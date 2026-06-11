#!/usr/bin/env python3
"""
src/forecast_hotspots.py — Short-horizon traffic-hotspot forecaster (ConvLSTM).

Consumes the grid-aligned synthetic traffic tensors produced by
``src/traffic_export.py`` (``(T, C, H, W)`` .npy on the Mollweide 1 km grid) and
learns a spatio-temporal forecaster: given the last ``L`` frames of cell-level
traffic, predict the next ``K`` frames.  Thresholding the prediction yields a
*future hotspot map* — the signal a LEO constellation would use to pre-emptively
schedule handovers before congestion peaks (the project's stated end-goal).

Model: a compact ConvLSTM encoder–decoder (one ConvLSTM cell each side, a 1×1
conv head), small enough to train on CPU over a city-sized ROI grid.

CLI
---
    # Train on one or more exported tensors
    python src/forecast_hotspots.py train \\
        --tensor data/outputs/simulation/new_delhi/dataset/traffic_tensor_baseline.npy \\
        --epochs 15 --seq-len 12 --horizon 3 --out data/outputs/forecast/new_delhi

    # Evaluate a checkpoint (MAE/RMSE + hotspot hit-rate vs persistence baseline)
    python src/forecast_hotspots.py evaluate --checkpoint <dir>/convlstm.pt

    # Predict next-K frames from the tail of a tensor
    python src/forecast_hotspots.py predict --checkpoint <dir>/convlstm.pt \\
        --tensor <...>.npy

Caveats
-------
A single city × scenario × ~300 ticks yields few training windows, so v1 is best
treated as a *validated demonstration* of the pre-emptive-handover concept.  For
a stronger model, train across multiple cities/scenarios and run the simulation
at ``--sample-every 1`` (denser frames) with several seeds.  No GPU is assumed;
keep the ROI grid modest and the model small.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # keep numpy helpers importable without torch
    TORCH_AVAILABLE = False


# ── data prep (pure numpy — testable without torch) ─────────────────────────────

def normalize_tensor(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel max-normalise a (T, C, H, W) tensor into [0, 1].
    Returns (normed float32, scale[C]) where scale is the per-channel divisor.
    """
    series = series.astype(np.float32)
    C = series.shape[1]
    scale = np.ones(C, dtype=np.float32)
    for c in range(C):
        m = float(series[:, c].max())
        scale[c] = m if m > 1e-6 else 1.0
    normed = series / scale[None, :, None, None]
    return normed, scale


def make_windows(series: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding windows over the time axis of a (T, C, H, W) tensor.

    Returns X (N, L, C, H, W) inputs and Y (N, K, H, W) targets, where the
    target is channel 0 (volume) of the next ``horizon`` frames.
    """
    T = series.shape[0]
    n = T - seq_len - horizon + 1
    if n <= 0:
        raise ValueError(
            f'tensor too short: T={T} needs > seq_len+horizon={seq_len + horizon}'
        )
    X = np.stack([series[i:i + seq_len] for i in range(n)]).astype(np.float32)
    Y = np.stack([series[i + seq_len:i + seq_len + horizon, 0] for i in range(n)]).astype(np.float32)
    return X, Y


def chronological_split(
    n: int, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple[slice, slice, slice]:
    """Chronological train/val/test slices over n windows (no shuffling)."""
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    n_train = max(1, n - n_val - n_test)
    return (slice(0, n_train),
            slice(n_train, n_train + n_val),
            slice(n_train + n_val, n))


def hit_rate(pred: np.ndarray, true: np.ndarray, top_n: int) -> float:
    """
    Fraction of the true top-N congested cells recovered by the predicted
    top-N, averaged over frames.  ``pred``/``true`` are (..., H, W).
    """
    pred = pred.reshape(-1, pred.shape[-2] * pred.shape[-1])
    true = true.reshape(-1, true.shape[-2] * true.shape[-1])
    top_n = max(1, min(top_n, pred.shape[1]))
    hits = []
    for p, t in zip(pred, true):
        if t.max() <= 0:
            continue
        p_top = set(np.argpartition(p, -top_n)[-top_n:].tolist())
        t_top = set(np.argpartition(t, -top_n)[-top_n:].tolist())
        hits.append(len(p_top & t_top) / top_n)
    return float(np.mean(hits)) if hits else float('nan')


def load_tensors(paths: list[str | Path]) -> list[np.ndarray]:
    out = []
    for p in paths:
        arr = np.load(p)
        if arr.ndim != 4:
            raise ValueError(f'{p}: expected (T,C,H,W), got {arr.shape}')
        out.append(arr.astype(np.float32))
    return out


# ── model + training (torch) ────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class ConvLSTMCell(nn.Module):
        """Standard convolutional LSTM cell (Shi et al., 2015)."""

        def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
            super().__init__()
            pad = kernel_size // 2
            self.hidden_dim = hidden_dim
            self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                                  kernel_size, padding=pad)

        def forward(self, x, state):
            h, c = state
            gates = self.conv(torch.cat([x, h], dim=1))
            i, f, o, g = torch.chunk(gates, 4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            return h, c

        def init_state(self, batch, shape, device):
            h = torch.zeros(batch, self.hidden_dim, *shape, device=device)
            c = torch.zeros(batch, self.hidden_dim, *shape, device=device)
            return h, c

    class ConvLSTMForecaster(nn.Module):
        """Encoder–decoder ConvLSTM: L input frames → K predicted volume frames."""

        def __init__(self, in_channels: int, hidden_dim: int = 32,
                     horizon: int = 3, kernel_size: int = 3):
            super().__init__()
            self.horizon  = horizon
            self.encoder  = ConvLSTMCell(in_channels, hidden_dim, kernel_size)
            self.decoder  = ConvLSTMCell(1, hidden_dim, kernel_size)
            self.head     = nn.Conv2d(hidden_dim, 1, 1)

        def forward(self, x):                       # x: (B, L, C, H, W)
            b, L, _, H, W = x.shape
            dev = x.device
            h, c = self.encoder.init_state(b, (H, W), dev)
            for t in range(L):
                h, c = self.encoder(x[:, t], (h, c))
            dh, dc = h, c
            inp = x[:, -1, 0:1]                     # last observed volume frame
            outs = []
            for _ in range(self.horizon):
                dh, dc = self.decoder(inp, (dh, dc))
                frame = self.head(dh)               # (B,1,H,W)
                outs.append(frame)
                inp = frame
            return torch.cat(outs, dim=1)           # (B, K, H, W)

    def _device() -> 'torch.device':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _to_loader(X, Y, batch_size, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def train_forecaster(
        tensors: list[np.ndarray],
        out_dir: Path,
        seq_len: int = 12,
        horizon: int = 3,
        hidden_dim: int = 32,
        epochs: int = 15,
        batch_size: int = 8,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Normalise each tensor with shared per-channel scale (from concatenation).
        stacked = np.concatenate(tensors, axis=0)
        _, scale = normalize_tensor(stacked)
        C, Hh, Ww = stacked.shape[1], stacked.shape[2], stacked.shape[3]

        Xs, Ys = [], []
        for arr in tensors:
            normed = arr / scale[None, :, None, None]
            X, Y = make_windows(normed, seq_len, horizon)
            Xs.append(X); Ys.append(Y)
        X = np.concatenate(Xs, axis=0); Y = np.concatenate(Ys, axis=0)

        tr, va, te = chronological_split(len(X))
        dev   = _device()
        model = ConvLSTMForecaster(C, hidden_dim, horizon).to(dev)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.MSELoss()

        train_loader = _to_loader(X[tr], Y[tr], batch_size, shuffle=True)
        val_loader   = _to_loader(X[va], Y[va], batch_size, shuffle=False)

        best_val, history = float('inf'), []
        ckpt_path = out_dir / 'convlstm.pt'
        for ep in range(1, epochs + 1):
            model.train()
            tl = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                tl += loss.item() * len(xb)
            tl /= max(1, len(X[tr]))

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    vl += lossf(model(xb), yb).item() * len(xb)
            vl /= max(1, len(X[va]))
            history.append({'epoch': ep, 'train_mse': tl, 'val_mse': vl})
            print(f'  epoch {ep:3d}/{epochs}  train_mse={tl:.5f}  val_mse={vl:.5f}')

            if vl < best_val:
                best_val = vl
                torch.save({
                    'model_state'  : model.state_dict(),
                    'config'       : {'in_channels': C, 'hidden_dim': hidden_dim,
                                      'horizon': horizon, 'seq_len': seq_len},
                    'scale'        : scale.tolist(),
                    'grid_shape'   : [int(Hh), int(Ww)],
                }, ckpt_path)

        # Hold-out evaluation on the test split.
        metrics = _evaluate_split(model, X[te], Y[te], scale[0], dev, out_dir,
                                  make_plots=True)
        metrics.update({'best_val_mse': best_val, 'n_windows': int(len(X)),
                        'epochs': epochs, 'seq_len': seq_len, 'horizon': horizon})
        (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'  Saved checkpoint -> {ckpt_path}')
        print(f'  Test MAE={metrics["mae"]:.4f}  RMSE={metrics["rmse"]:.4f}  '
              f'hit-rate@{metrics["top_n"]}={metrics["hit_rate"]:.3f} '
              f'(persistence {metrics["hit_rate_persistence"]:.3f})')
        return metrics

    def _evaluate_split(model, X, Y, vol_scale, dev, out_dir, make_plots=False,
                        top_n=20):
        model.eval()
        preds = []
        with torch.no_grad():
            loader = _to_loader(X, Y, 8, shuffle=False)
            for xb, _ in loader:
                preds.append(model(xb.to(dev)).cpu().numpy())
        pred = np.concatenate(preds, axis=0) if preds else np.zeros_like(Y)

        # Denormalise the volume channel back to agent-crossings.
        pred_d = pred * vol_scale
        true_d = Y * vol_scale
        persistence = X[:, -1, 0] * vol_scale       # last observed frame, repeated implicitly

        mae  = float(np.mean(np.abs(pred_d - true_d)))
        rmse = float(np.sqrt(np.mean((pred_d - true_d) ** 2)))
        hr   = hit_rate(pred_d[:, 0], true_d[:, 0], top_n)
        hr_p = hit_rate(np.repeat(persistence[:, None], 1, axis=1)[:, 0],
                        true_d[:, 0], top_n)

        if make_plots and len(pred_d):
            _save_prediction_pngs(X, true_d, pred_d, vol_scale, out_dir)

        return {'mae': mae, 'rmse': rmse, 'hit_rate': hr,
                'hit_rate_persistence': hr_p, 'top_n': top_n,
                'n_test_windows': int(len(X))}

    def _save_prediction_pngs(X, true_d, pred_d, vol_scale, out_dir, n=3):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        idxs = np.linspace(0, len(pred_d) - 1, min(n, len(pred_d))).astype(int)
        for k, i in enumerate(idxs):
            last_in = X[i, -1, 0] * vol_scale
            vmax = max(float(true_d[i, 0].max()), float(pred_d[i, 0].max()), 1.0)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
            for ax, img, title in [
                (axes[0], last_in,     'Last observed (t)'),
                (axes[1], true_d[i, 0], 'Actual (t+1)'),
                (axes[2], pred_d[i, 0], 'Predicted (t+1)'),
            ]:
                im = ax.imshow(img, cmap='hot_r', vmin=0, vmax=vmax)
                ax.set_title(title, fontsize=10); ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.suptitle('ConvLSTM traffic-hotspot forecast', fontsize=12)
            fig.savefig(Path(out_dir) / f'forecast_sample_{k}.png', dpi=120)
            plt.close(fig)

    def load_checkpoint(path: str | Path):
        ckpt = torch.load(path, map_location=_device())
        cfg = ckpt['config']
        model = ConvLSTMForecaster(cfg['in_channels'], cfg['hidden_dim'],
                                   cfg['horizon']).to(_device())
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model, ckpt


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise SystemExit(
            'PyTorch is required for forecasting. Install it with:\n'
            '    pip install torch\n'
            '(see requirements.txt).'
        )


def do_train(args: argparse.Namespace) -> None:
    _require_torch()
    tensors = load_tensors(args.tensor)
    out_dir = Path(args.out) if args.out else Path('data/outputs/forecast/default')
    print(f'Training ConvLSTM on {len(tensors)} tensor(s), '
          f'shapes={[t.shape for t in tensors]}')
    train_forecaster(
        tensors, out_dir,
        seq_len=args.seq_len, horizon=args.horizon, hidden_dim=args.hidden_dim,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )
    print(f'All forecast outputs written to: {out_dir}')


def do_evaluate(args: argparse.Namespace) -> None:
    _require_torch()
    model, ckpt = load_checkpoint(args.checkpoint)
    cfg     = ckpt['config']
    scale   = np.array(ckpt['scale'], dtype=np.float32)
    tensors = load_tensors(args.tensor) if args.tensor else None
    if tensors is None:
        raise SystemExit('evaluate needs --tensor to score against.')
    out_dir = Path(args.checkpoint).parent
    Xs, Ys = [], []
    for arr in tensors:
        normed = arr / scale[None, :, None, None]
        X, Y = make_windows(normed, cfg['seq_len'], cfg['horizon'])
        Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs, axis=0); Y = np.concatenate(Ys, axis=0)
    _, _, te = chronological_split(len(X))
    metrics = _evaluate_split(model, X[te], Y[te], scale[0], _device(), out_dir,
                              make_plots=True)
    print(json.dumps(metrics, indent=2))


def do_predict(args: argparse.Namespace) -> None:
    _require_torch()
    model, ckpt = load_checkpoint(args.checkpoint)
    cfg   = ckpt['config']
    scale = np.array(ckpt['scale'], dtype=np.float32)
    arr   = load_tensors([args.tensor])[0]
    normed = arr / scale[None, :, None, None]
    L = cfg['seq_len']
    if arr.shape[0] < L:
        raise SystemExit(f'tensor has {arr.shape[0]} frames, need >= seq_len={L}')
    x = torch.from_numpy(normed[-L:][None]).float().to(_device())
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0] * scale[0]
    out_path = Path(args.out) if args.out else Path(args.tensor).with_name('prediction.npy')
    np.save(out_path, pred.astype(np.float32))
    print(f'Predicted next {cfg["horizon"]} frame(s) {pred.shape} -> {out_path}')


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='forecast_hotspots',
        description='ConvLSTM short-horizon traffic-hotspot forecaster.',
    )
    sub = p.add_subparsers(dest='command', required=True)

    tr = sub.add_parser('train', help='Train a ConvLSTM forecaster on traffic tensors.')
    tr.add_argument('--tensor', nargs='+', required=True,
                    help='One or more traffic_tensor_*.npy paths.')
    tr.add_argument('--out', default=None, help='Output dir for checkpoint/metrics/PNGs.')
    tr.add_argument('--seq-len',   type=int,   default=12)
    tr.add_argument('--horizon',   type=int,   default=3)
    tr.add_argument('--hidden-dim', type=int,  default=32)
    tr.add_argument('--epochs',    type=int,   default=15)
    tr.add_argument('--batch-size', type=int,  default=8)
    tr.add_argument('--lr',        type=float, default=1e-3)
    tr.add_argument('--seed',      type=int,   default=42)

    ev = sub.add_parser('evaluate', help='Evaluate a checkpoint on tensor(s).')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--tensor', nargs='+', required=True)

    pr = sub.add_parser('predict', help='Predict next-K frames from a tensor tail.')
    pr.add_argument('--checkpoint', required=True)
    pr.add_argument('--tensor', required=True)
    pr.add_argument('--out', default=None)

    return p


def main() -> None:
    args = _make_parser().parse_args()
    if args.command == 'train':
        do_train(args)
    elif args.command == 'evaluate':
        do_evaluate(args)
    else:
        do_predict(args)


if __name__ == '__main__':
    main()
