# LEO-Traffic-Modelling

This repository contains a geospatial traffic-synthesis pipeline built to support preemptive handover optimization in LEO satellite constellations. The core idea is to start from population-density data, convert that into city-scale hotspot demand, attach that demand to a real urban road network, and simulate how traffic loads evolve under congestion and disruption scenarios. The final outputs are both visual and machine-readable: comparative traffic analytics, basemap-based animations, and grid-aligned traffic tensors that can be reused in downstream modeling workflows.

## Project Objective

The project aims to generate realistic synthetic urban traffic patterns for selected cities in a way that is spatially consistent, reproducible, and useful for downstream network-planning tasks. Instead of beginning directly with a traffic simulator, the implementation first constructs a reliable geospatial demand surface, then narrows that surface to city-scale regions of interest, then identifies the most intense subregions as hotspots, and only then translates that demand onto a road network for simulation. This design keeps the simulation anchored to observable geographic structure rather than arbitrary source distributions.

## Methodology

The implementation follows a staged pipeline.

First, the baseline demand surface is built by fusing global WorldPop and LandScan rasters into a shared equal-area Mollweide grid. Both rasters are reprojected into the same spatial frame, normalized in log space using percentile bounds, and combined into a single float32 baseline raster. The key decision here is to keep the entire project centered on `ESRI:54009` so that later hotspot analysis, network snapping, and traffic export all remain spatially aligned.

Second, the project extracts city-level regions of interest from UCDB urban polygons. Selected cities are matched against the UCDB layer, buffered to capture surrounding urban context, and used to clip per-city baseline rasters and binary masks. This step turns a global demand raster into compact city-specific analysis windows while preserving the same grid geometry and resolution.

Third, hotspot demand is derived from the clipped city raster. The implementation transforms near-saturated baseline values into a hotspot score, smooths that score with local neighborhood aggregation, selects the upper-quantile candidate cells, and clusters them into weighted hotspot centers. The purpose of this stage is to compress a dense raster field into a smaller, structured set of high-value demand anchors before moving to the road-network model.

Fourth, hotspot demand is attached to the urban road graph. The project downloads OSM drive networks for each city ROI, projects them into the same Mollweide CRS, snaps hotspot centers to nearest road nodes, and aggregates those snapped centers into reusable node-level demand tables. This creates a graph-native demand representation and a city bundle containing the network, hotspot mappings, spatial layers, and manifest metadata.

Fifth, the project runs a congestion-aware agent-based traffic simulation on the saved city bundle. Agents spawn from hotspot-weighted origin nodes, select hotspot destinations, and move through the road network using Dijkstra-based routing trees that are periodically rebuilt as congestion changes. The simulator applies BPR-style dynamic costs, supports diurnal demand modulation, and compares a baseline run against stress scenarios such as capacity drops, edge closures, and hotspot surges. This is where the project turns static demand structure into dynamic traffic flow.

Sixth, the simulated traffic is exported in two forms. For analysis, the project writes scenario metrics, utilization plots, corridor heatmaps, flow-delta views, and animated traffic visualizations. For reuse in downstream modeling, it converts edge-level flow histories into grid-aligned spatio-temporal tensors registered to the same city ROI raster grid, producing `NumPy` arrays and `GeoTIFF` outputs that preserve spatial consistency across the full pipeline.

Finally, the repository also includes a map-overlay workflow that renders one simulation scenario directly on top of warped OSM basemap tiles. This makes the output easier to inspect visually by showing utilization changes in the exact urban geography where they occur.

## Tools And Libraries

The implementation is built primarily in Python and relies on a geospatial and scientific-computing stack:

- `Rasterio` for raster inspection, reprojection, clipping, and GeoTIFF export
- `GeoPandas` for vector IO, ROI handling, and spatial layer export
- `Shapely` for polygon buffering, geometry construction, and spatial transformations
- `NumPy` for array processing and tensor construction
- `Pandas` for tabular demand aggregation, manifests, and simulation metrics
- `scikit-learn` for hotspot clustering with `MiniBatchKMeans`
- `OSMnx` for downloading and exporting urban drive networks
- `NetworkX` for graph construction and route representation
- `SciPy` for sparse Dijkstra routing on large road graphs
- `Matplotlib` for plots and GIF animations
- `Requests`, `xyzservices`, and `Pillow` for web-tile basemap fetching, warping, and image assembly

## Main Outputs

The tracked source code produces several artifact types across the pipeline:

- fused baseline rasters and normalization manifests
- clipped city ROI rasters and masks
- hotspot summaries, centers, and snapped-node mappings
- projected road-network bundles in `GraphML` and `GeoPackage` format
- simulation metrics and comparative visual reports
- grid-aligned traffic tensors in `NumPy` and `GeoTIFF` form
- basemap-backed traffic flow animations

## Final Simulation Animation

The animation below is the final Bengaluru capacity-drop map-overlay simulation generated by the repository pipeline.

![Bengaluru Capacity Drop Simulation](data/outputs/map_overlay/bengaluru/animation/traffic_flow_capacity_drop.gif)
