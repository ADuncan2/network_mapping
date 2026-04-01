import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime

log_dir = "logs"

# Regex patterns
timestamp_fid_pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?FID:(\d+)"
)
dfs_pattern = re.compile(r"DFS finished with \d+ nodes, (\d+) service points, \d+ edges")
duration_pattern = re.compile(r"Processing duration:\s+(\d+):(\d+):(\d+)\.(\d+)")

records = []

all_logs = [f for f in os.listdir(log_dir) if f.endswith(".log")]
print(f"Found {len(all_logs)} log files in folder.")

for filename in all_logs:
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "r") as f:
        content = f.read()

    # Extract timestamps + FIDs
    ts_fid_pairs = timestamp_fid_pattern.findall(content)
    # -> list of (timestamp_string, fid_string)

    dfs_matches = dfs_pattern.findall(content)
    dur_matches = duration_pattern.findall(content)

    # Align based on order
    for (timestamp_str, fid), sp, dur in zip(ts_fid_pairs, dfs_matches, dur_matches):

        # Parse timestamp
        mapped_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

        service_points = int(sp)
        hours, minutes, seconds, micros = map(int, dur)
        duration = timedelta(
            hours=hours, minutes=minutes, seconds=seconds, microseconds=micros
        ).total_seconds() / 60.0  # minutes

        records.append({
            "fid": int(fid),
            "mapped_time": mapped_time,
            "service_points": service_points,
            "duration_minutes": duration,
            "logfile": filename
        })

# Build dataframe
df = pd.DataFrame(records)

# Remove duplicate FIDs
df = df.drop_duplicates(subset=['fid'])

# Sort by time (important for line plots)
df = df.sort_values("mapped_time").reset_index(drop=True)

print(f"Extracted {len(df)} FID records from {len(all_logs)} log files.")
print(df.head())
print(df.describe())



# # Plot Duration vs Service Points
# plt.figure(figsize=(8,6))
# plt.scatter(df["service_points"], df["duration_minutes"], alpha=0.7, edgecolor="k")
# plt.xlabel("Number of Service Points")
# plt.ylabel("Processing Duration (minutes)")
# plt.title("Mapping Time vs Network Size (per FID)")
# plt.grid(True)
# plt.show()

# Plot Duration over Time
plt.figure(figsize=(12, 5))
plt.plot(df["mapped_time"], df["duration_minutes"], marker="o")
plt.xlabel("Mapping Time")
plt.ylabel("Processing Duration (minutes)")
plt.title("Network Mapping Duration Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()