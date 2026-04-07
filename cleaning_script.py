"""
Preprocessing pipeline for the Gridstock network mapping project.

Runs all data preparation steps in order. Each step can be commented
out individually if it has already been run.  A preprocessing report
is written to data/cleaning_report.txt at the end.

Pipeline order:
  1. create_lv_db          — Import 15 layers from assets.gpkg into lv_assets.sqlite
  2. create_conn_comp      — Fresh indexed copy of conn → conn_comp in network_data.sqlite
  3. merge_service_points  — Collapse collocated service points / joints in conn_comp
  4. simplify_substations  — Collapse internal substation nodes to boundary FID
  5. simplify_pmt_substations — Create 5m buffer circles for PMTs, find flux lines
  6. enrich_substations    — Spatial join: transformer capacity → General Boundary rows
  7. copy_conn_to_lv       — Copy conn_comp (fid_from, fid_to, flow) into lv_assets.sqlite
  8. create_graph_db       — Create empty output database (results/graph.sqlite)

NOT INCLUDED (removed from previous pipeline):
  - collapse_switch_boxes  — No longer needed; Way status checked directly in DFS
  - creating_conn_new      — No longer needed; conn_comp created fresh from conn
"""

import os

from gridstock.dbcleaning import (
    PreprocessingReport,
    create_lv_db,
    create_conn_comp,
    merge_service_points,
    simplify_substations,
    simplify_pmt_substations,
    enrich_substations,
    copy_conn_to_lv,
    create_graph_db,
)

# Initialise preprocessing report
report = PreprocessingReport()

# Step 1: Create lv_assets.sqlite from 15 layers of assets.gpkg
print("=" * 60)
print("Step 1: Creating lv_assets.sqlite...")
print("=" * 60)
create_lv_db(report)
print("done\n")

# Step 2: Create conn_comp as fresh indexed copy of conn
print("=" * 60)
print("Step 2: Creating conn_comp...")
print("=" * 60)
create_conn_comp(report)
print("done\n")

# Step 3: Merge collocated service points and joints
print("=" * 60)
print("Step 3: Merging service points...")
print("=" * 60)
merge_service_points("data/assets.gpkg", report)
print("done\n")

# Step 4: Simplify substations (ground-mounted)
print("=" * 60)
print("Step 4: Simplifying substations...")
print("=" * 60)
simplify_substations(report)
print("done\n")

# Step 5: Simplify PMT substations (pole-mounted)
print("=" * 60)
print("Step 5: Simplifying PMT substations...")
print("=" * 60)
simplify_pmt_substations(report)
print("done\n")

# Step 6: Enrich substations with transformer capacity
print("=" * 60)
print("Step 6: Enriching substations...")
print("=" * 60)
enrich_substations(report)
print("done\n")

# Step 7: Copy conn_comp into lv_assets.sqlite
print("=" * 60)
print("Step 7: Copying conn_comp to lv_assets.sqlite...")
print("=" * 60)
copy_conn_to_lv(report)
print("done\n")

# Step 8: Create empty results/graph.sqlite
print("=" * 60)
print("Step 8: Creating results/graph.sqlite...")
print("=" * 60)
os.makedirs("results", exist_ok=True)
create_graph_db()
print("done\n")

# Write report
print("=" * 60)
print("Writing preprocessing report...")
print("=" * 60)
report.write()
