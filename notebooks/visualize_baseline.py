from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a GeoTIFF raster with safe downsampling.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/baseline_mollweide_1km_2022.tif"),
        help="Path to the GeoTIFF to visualize.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2000,
        help="Maximum displayed width or height in pixels after downsampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG path. Defaults to the TIFF path with a .png suffix.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the PNG without opening an interactive Matplotlib window.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    tif_path = args.input.expanduser().resolve()
    output_path = (args.output or tif_path.with_suffix(".png")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        scale = max(src.width / args.max_size, src.height / args.max_size, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))

        data = src.read(
            1,
            masked=True,
            out_shape=(out_height, out_width),
        )
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    values = data.compressed()
    if values.size == 0:
        raise SystemExit(f"No valid pixels found in {tif_path}")

    vmin, vmax = np.percentile(values, [2, 98])

    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(data, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Normalized Baseline Intensity")

    ax.set_title(tif_path.name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(output_path)

    if args.no_show:
        plt.close(fig)
        return

    plt.show()


if __name__ == "__main__":
    main()
