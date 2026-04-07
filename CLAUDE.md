# CLAUDE.md

## Project Overview

**Gridstock** maps UK electricity distribution networks (ENWL — Electricity North West Limited) to buildings using GIS data. The pipeline discovers low-voltage (LV) network topology via depth-first search from substations, stores it in SQLite, then matches service point nodes to building geometries.

## Architecture

### Pipeline stages

1. **Preprocessing** (`cleaning_script.py`) — One-time setup: creates `conn_comp` indexed table, simplifies substations/service points, builds `lv_assets.sqlite` and empty `results/graph.sqlite`.
2. **Network mapping** (`single_fid_mapping.py`) — Multiprocessing pipeline that iterates over substation FIDs from `lv_assets.sqlite`, runs DFS via `map_substation()`, and writes topology to `results/graph.sqlite`. Uses batched logging and SQLite WAL mode.
3. **Building matching** (`building_matching.py`) — Loads mapped networks from `results/graph.sqlite` as NetworkX graphs via `MappedNetwork`, finds nearby buildings from parquet, matches service points to buildings using overlap → buffer spatial joins.

### Key modules (`gridstock/`)

| File | Purpose |
|------|---------|
| `mapfuncs.py` | DFS search functions, coordinate interpolation, `map_substation()` |
| `recorder.py` | `NetworkData` class — stores DFS results, writes to `results/graph.sqlite` and `results/summary.csv` |
| `dbcleaning.py` | DB setup: `create_lv_db()`, `create_graph_db()`, substation/switch simplification |
| `network_loader.py` | `MappedNetwork` class — loads mapped networks from `results/graph.sqlite` into NetworkX |

### Validation scripts (`validation/`)

| File | Purpose |
|------|---------|
| `benchmark_dfs.py` | Per-stage timing benchmarks across 40 substations |
| `diagnose_dfs.py` | Large-scale diagnostic runner with SQLite results DB and resume support |

### Input data (in `data/`, gitignored)

- `assets.gpkg` — Source GeoPackage with all network assets (layers: LV Conductor, Service Point, General Boundary, etc.)
- `network_data.sqlite` — Raw connectivity table (`conn`)
- `lv_assets.sqlite` — Filtered LV-only subset of assets
- `all_buildings_in_enwl_27700.parquet` — Building footprints with UPRN/TOID in EPSG:27700

### Output data (in `results/`, gitignored)

- `graph.sqlite` — Mapped network incidence lists, edge/node properties, mapped substations (WAL mode)
- `summary.csv` — Per-substation mapping summary stats
- `temp/` — Per-worker flux line databases and validation temp files

## Tech Stack

- **Python 3.11** (uv for dependency management)
- **Spatial**: GeoPandas, Shapely, Fiona, PyProj (CRS: EPSG:27700 for data, 3857/4326 for visualization)
- **Graph**: NetworkX
- **Storage**: SQLite (WAL mode for concurrent writes), GeoParquet
- **Viz**: Plotly (interactive maps)
- **Other**: SciPy, tqdm, multiprocessing

## Development Commands

```bash
# Install dependencies
uv sync

# Run preprocessing (one-time)
uv run python cleaning_script.py

# Step 1: Run network mapping (multiprocessing)
uv run python single_fid_mapping.py

# Step 2: Run building matching
uv run python building_matching.py
```

## Conventions

- Substation FIDs are integers (e.g., `11391343`).
- Geometries are stored as WKB blobs in SQLite; use `shapely.wkb.loads()` / `from_wkb()` to deserialize.
- `data/` and `results/` are gitignored — never commit data files.
- The recursion limit is set to 5000 in `mapfuncs.py` for DFS.
- DFS budget: 50,000 calls or 120s per substation (`DFSBudgetExceeded`).

## Obsidian Permissions

- You MAY use the Write tool to create new notes in: `C:\Users\ucbvaad\Documents\My notes vault\Work notes\Claude notes\`
- You MAY NOT edit, move, or delete existing vault notes
- When creating notes, use the work note template style from the vault
