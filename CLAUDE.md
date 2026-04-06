# CLAUDE.md

## Project Overview

**Gridstock** maps UK electricity distribution networks (ENWL — Electricity North West Limited) to buildings using GIS data. The pipeline discovers low-voltage (LV) network topology via depth-first search from substations, stores it in SQLite, then matches service point nodes to building geometries.

## Architecture

### Pipeline stages

1. **Preprocessing** (`cleaning_script.py`) — One-time setup: creates `conn_new` indexed table, simplifies substations/switch boxes/service points, builds `lv_assets.sqlite` and empty `graph.sqlite`.
2. **Network mapping** (`single_fid_mapping.py`) — Multiprocessing pipeline that iterates over substation FIDs from `data/assets.gpkg`, runs DFS via `map_substation()`, and writes results to `data/graph.sqlite`. Uses batched logging and SQLite WAL mode.
3. **Building matching** (`building_matching.py`, `building_match_vis.py`) — Loads mapped networks from `graph.sqlite` as NetworkX graphs via `DistributionNetwork`, finds nearby buildings from parquet, matches service points to buildings using overlap → buffer → nearest-edge spatial joins.

### Key modules (`gridstock/`)

| File | Purpose |
|------|---------|
| `mapfuncs.py` | DFS search functions, coordinate interpolation |
| `recorder.py` | `NetworkData` class — stores DFS results, writes to `graph.sqlite` and `summary.csv` |
| `dbcleaning.py` | DB setup: `create_lv_db()`, `create_graph_db()`, substation/switch simplification |
| `network_parsing_EEA.py` | `DistributionNetwork` class — loads networks from SQLite into NetworkX/pandapower |
| `network_parsing.py` | Earlier version of network parsing |
| `plotting.py` | Matplotlib geometry plotting (singledispatch for Point/LineString/Polygon) |
| `config_manager.py` | YAML config loader with hash-based change detection |
| `creating_conn_new.py` | Creates indexed clone of `conn` table for faster queries |

### Data files (in `data/`, gitignored)

- `assets.gpkg` — Source GeoPackage with all network assets (layers: LV Conductor, Service Point, General Boundary, etc.)
- `network_data.sqlite` — Raw connectivity table (`conn`)
- `lv_assets.sqlite` — Filtered LV-only subset of assets
- `graph.sqlite` — Output: mapped network incidence lists, edge/node properties, mapped substations (WAL mode)
- `all_buildings_in_enwl_27700.parquet` — Building footprints with UPRN/TOID in EPSG:27700
- `summary.csv` — Per-substation mapping summary stats

## Tech Stack

- **Python 3.11** (Poetry for dependency management)
- **Spatial**: GeoPandas, Shapely, Fiona, PyProj (CRS: EPSG:27700 for data, 3857/4326 for visualization)
- **Graph**: NetworkX, pandapower
- **Storage**: SQLite (WAL mode for concurrent writes), GeoParquet
- **Viz**: Plotly (interactive maps), Matplotlib/Seaborn (static plots)
- **Other**: OSMnx, SciPy, tqdm, multiprocessing

## Development Commands

```bash
# Install dependencies
poetry install

# Run network mapping (multiprocessing)
poetry run python single_fid_mapping.py

# Run building matching (single FID for debugging)
poetry run python building_matching.py

# Run building match visualization
poetry run python building_match_vis.py
```

## Conventions

- Substation FIDs are integers (e.g., `11391343`).
- Geometries are stored as WKB blobs in SQLite; use `shapely.wkb.loads()` / `from_wkb()` to deserialize.
- The `temp/` directory contains experimental/one-off scripts — not part of the main pipeline.
- Problem substations are hardcoded exclusion lists in `single_fid_mapping.py`.
- `data/` and `results/` are gitignored — never commit data files.

## Known Issues

- Some substations crash the mapper and are excluded via `problem_subs` list in `single_fid_mapping.py`.
- `building_matching.py` and `building_match_vis.py` have diverged — `building_match_vis.py` has a 3rd matching pass (nearest edge within 10m) that `building_matching.py` lacks.
- The recursion limit is set to 5000 in `mapfuncs.py` for DFS.

## Obsidian Permissions

- You MAY use the Write tool to create new notes in: `C:\Users\ucbvaad\Documents\My notes vault\Work notes\Claude notes\`
- You MAY NOT edit, move, or delete existing vault notes
- When creating notes, use the work note template style from the vault
