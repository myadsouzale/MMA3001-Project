
"""
Performance Profiling of the Occupancy Processing Pipeline
==========================================================

Measure the runtime and memory consumption of the two main
data-processing stages in the building occupancy project:

1. Cleaning the raw occupancy sensor dataset.
2. Aggregating cleaned observations into hourly occupancy.

The experiment evaluates the computational resources required
to process a large historical occupancy dataset containing
approximately nine million sensor observations.

Methodology
-----------
Each processing script is launched as a separate Python
subprocess using the current Python interpreter.

The profiler measures:

- Total elapsed execution time using time.perf_counter().
- Peak sampled resident set size (RSS) in megabytes.
- Process exit code to identify unsuccessful executions.

Memory usage is sampled approximately every 0.1 seconds
using psutil.

The reported memory value includes the parent subprocess
and any child processes detected during sampling.

If a processing stage fails, the profiler records the
failure and stops before executing the next stage.

Inputs
------
src/clean_occupancy_data.py
    Cleans raw occupancy sensor observations.

src/aggregate_occupancy_hourly.py
    Aggregates cleaned observations into hourly data.

Outputs
-------
results/optimisation/pipeline_performance.csv
    Contains runtime, peak sampled RSS and exit code
    for each executed processing stage.

Assumptions
-----------
- Both processing scripts can be run independently
  from the project root directory.
- An exit code of zero indicates successful execution.
- Sampled RSS provides an approximate measure of
  the memory required by each processing stage.
- The same Python environment is used for profiling
  and normal project execution.

Limitations
-----------
Memory is sampled at discrete intervals, so short-lived
memory peaks between samples may not be recorded.

RSS measures resident physical memory rather than total
allocated or reserved virtual memory.

Memory reported for a process and its children is summed,
which may count some shared physical memory more than once.

Runtime measurements depend on computer hardware,
disk performance and background system activity.

The profiling experiment runs the actual processing
scripts and therefore regenerates their output datasets.

Usage
-----
Run from the project root directory:

    python -u src/profile_pipeline.py

Requirements
------------
The psutil package must be installed:

    pip install psutil
"""

import csv
import subprocess
import sys
import time

from pathlib import Path

import psutil


# ==========================================
# 1. FILE PATHS AND SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    ROOT
    / "results"
    / "optimisation"
)

SCRIPTS = [
    "clean_occupancy_data.py",
    "aggregate_occupancy_hourly.py"
]

MEMORY_SAMPLE_INTERVAL = 0.1

MB_CONVERSION = 1024 ** 2

CSV_FIELDS = [
    "Script",
    "Runtime (s)",
    "Peak Sampled RSS (MB)",
    "Exit Code"
]


# ==========================================
# 2. MEASURE PROCESS MEMORY
# ==========================================

def get_process_memory_mb(process_id):
    """
    Measure combined resident memory for a process tree.

    The function obtains the resident set size (RSS)
    of a process and its detected child processes.

    Parameters
    ----------
    process_id : int
        Operating-system process identifier.

    Returns
    -------
    float
        Combined resident memory in megabytes.

    Notes
    -----
    Processes that terminate during measurement
    or cannot be accessed are skipped.

    Shared physical memory may be counted more
    than once when summing multiple processes.
    """

    try:

        parent = psutil.Process(
            process_id
        )

        processes = (
            [parent]
            + parent.children(recursive=True)
        )

        total_rss = 0

        for proc in processes:

            try:

                total_rss += (
                    proc.memory_info().rss
                )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied
            ):

                pass

        return (
            total_rss / MB_CONVERSION
        )

    except psutil.NoSuchProcess:

        return 0.0


# ==========================================
# 3. PROFILE ONE PROCESSING SCRIPT
# ==========================================

def profile_script(script_name):
    """
    Measure runtime and peak sampled memory of one script.

    The selected script is executed as a separate Python
    subprocess.

    Memory usage is sampled periodically until the
    subprocess terminates.

    Parameters
    ----------
    script_name : str
        Filename of the Python script located in src/.

    Returns
    -------
    dict
        Profiling results containing:

        - Script
        - Runtime (s)
        - Peak Sampled RSS (MB)
        - Exit Code

    Notes
    -----
    The subprocess uses sys.executable to ensure that
    the same Python interpreter is used.

    This function executes the target script and may
    overwrite datasets produced by that script.
    """

    script = (
        ROOT
        / "src"
        / script_name
    )

    print(
        f"\nProfiling {script_name}...",
        flush=True
    )

    start = time.perf_counter()

    # Run the script as a separate process.

    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(script)
        ],
        cwd=ROOT
    )

    peak_memory_mb = 0.0

    # --------------------------------------
    # Monitor memory while process runs
    # --------------------------------------

    while process.poll() is None:

        memory_mb = get_process_memory_mb(
            process.pid
        )

        peak_memory_mb = max(
            peak_memory_mb,
            memory_mb
        )

        time.sleep(
            MEMORY_SAMPLE_INTERVAL
        )

    # --------------------------------------
    # Record final process results
    # --------------------------------------

    return_code = process.wait()

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"Runtime: {elapsed:.2f} seconds"
    )

    print(
        f"Peak sampled RSS: "
        f"{peak_memory_mb:.2f} MB"
    )

    print(
        f"Exit code: {return_code}"
    )

    return {
        "Script": script_name,
        "Runtime (s)": elapsed,
        "Peak Sampled RSS (MB)": peak_memory_mb,
        "Exit Code": return_code
    }


# ==========================================
# 4. SAVE PROFILING RESULTS
# ==========================================

def save_results(results, output_file):
    """
    Save pipeline profiling results to a CSV file.

    Parameters
    ----------
    results : list of dict
        Profiling measurements for executed scripts.

    output_file : pathlib.Path
        Destination CSV file.

    Returns
    -------
    None
        Results are written to the specified CSV file.

    Notes
    -----
    The output includes column headings even when
    no profiling results are available.
    """

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ==========================================
# 5. MAIN PROFILING FUNCTION
# ==========================================

def main():
    """
    Profile the occupancy data-processing pipeline.

    Executes the cleaning and hourly aggregation
    scripts sequentially.

    Measures execution time, peak sampled RSS and
    process exit codes for each stage.

    If a script fails, subsequent stages are skipped.

    Returns
    -------
    None
        Results are saved to pipeline_performance.csv
        and printed to the terminal.

    Notes
    -----
    Running this function executes the complete
    cleaning and aggregation pipeline.

    Existing processed datasets may be overwritten.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    # ==========================================
    # 6. PROFILE EACH PROCESSING STAGE
    # ==========================================

    for script_name in SCRIPTS:

        result = profile_script(
            script_name
        )

        results.append(
            result
        )

        # Stop if a processing stage fails.

        if result["Exit Code"] != 0:

            print(
                "A script failed. "
                "Stopping profiling."
            )

            break

    # ==========================================
    # 7. SAVE RESULTS
    # ==========================================

    output_file = (
        OUTPUT_DIR
        / "pipeline_performance.csv"
    )

    save_results(
        results,
        output_file
    )

    # ==========================================
    # 8. PRINT FINAL SUMMARY
    # ==========================================

    print("\n===================================")
    print("PIPELINE PERFORMANCE")
    print("===================================")

    for result in results:

        print(result)

    print(
        "\nSaved to:",
        output_file
    )


# ==========================================
# 9. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
