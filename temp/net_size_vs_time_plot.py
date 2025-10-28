import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

log_dir = "logs"

# Regex patterns
fid_pattern = re.compile(r"FID:(\d+)")
dfs_pattern = re.compile(r"DFS finished with \d+ nodes, (\d+) service points, \d+ edges")
duration_pattern = re.compile(r"Processing duration:\s+(\d+):(\d+):(\d+)\.(\d+)")

records = []

all_logs = [f for f in os.listdir(log_dir) if f.endswith(".log")]
print(f"Found {len(all_logs)} log files in folder.")

for filename in all_logs:
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "r") as f:
        content = f.read()

    # Find all FIDs in this file
    fids = fid_pattern.findall(content)
    dfs_matches = dfs_pattern.findall(content)
    dur_matches = duration_pattern.findall(content)

    # Sanity check: all three lists should align in length
    for fid, sp, dur in zip(fids, dfs_matches, dur_matches):
        service_points = int(sp)
        hours, minutes, seconds, micros = map(int, dur)
        duration = timedelta(
            hours=hours, minutes=minutes, seconds=seconds, microseconds=micros
        ).total_seconds() / 60.0  # minutes

        records.append({
            "fid": int(fid),
            "service_points": service_points,
            "duration_minutes": duration,
            "logfile": filename
        })

# Build dataframe
df = pd.DataFrame(records)

# Remove duplicate FIDs
df = df.drop_duplicates(subset=['fid'])

print(f"Extracted {len(df)} FID records from {len(all_logs)} log files.")
print(df.head())
print(df.describe())



# Plot
plt.figure(figsize=(8,6))
plt.scatter(df["service_points"], df["duration_minutes"], alpha=0.7, edgecolor="k")
plt.xlabel("Number of Service Points")
plt.ylabel("Processing Duration (minutes)")
plt.title("Mapping Time vs Network Size (per FID)")
plt.grid(True)
plt.show()
