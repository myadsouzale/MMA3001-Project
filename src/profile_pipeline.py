
import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results" / "optimisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "clean_occupancy_data.py",
    "aggregate_occupancy_hourly.py"
]

results = []

for script_name in SCRIPTS:
    script = ROOT / "src" / script_name

    print(f"\nProfiling {script_name}...", flush=True)

    start = time.perf_counter()

    # Run the script as a separate process.
    process = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        cwd=ROOT
    )

    peak_memory_mb = 0.0

    while process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            processes = [parent] + parent.children(recursive=True)

            total_rss = 0

            for proc in processes:
                try:
                    total_rss += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            peak_memory_mb = max(
                peak_memory_mb,
                total_rss / (1024 ** 2)
            )

        except psutil.NoSuchProcess:
            pass

        time.sleep(0.1)

    return_code = process.wait()
    elapsed = time.perf_counter() - start

    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Peak sampled RSS: {peak_memory_mb:.2f} MB")
    print(f"Exit code: {return_code}")

    results.append({
        "Script": script_name,
        "Runtime (s)": elapsed,
        "Peak Sampled RSS (MB)": peak_memory_mb,
        "Exit Code": return_code
    })

    if return_code != 0:
        print("A script failed. Stopping profiling.")
        break

output_file = OUTPUT_DIR / "pipeline_performance.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n===================================")
print("PIPELINE PERFORMANCE")
print("===================================")

for result in results:
    print(result)

print("\nSaved to:", output_file)
