"""
run_pipeline.py

Runs the full data pipeline (scripts 01-07) in order.
Stops immediately if any step fails.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

STEPS = [
    ("01_fetch_gst.py", "Parse GST Excel files"),
    ("02_fetch_electricity.py", "Parse POSOCO electricity CSV"),
    ("03_fetch_rbi.py", "Parse RBI credit Excel"),
    ("04_fetch_epfo.py", "Parse EPFO payroll Excel"),
    ("05_clean_and_merge.py", "Merge all 4 components"),
    ("06_compute_index.py", "Compute z-scores and composite index"),
    ("07_generate_json.py", "Generate JSON for dashboard"),
]


def main():
    print("=" * 70)
    print("Li Keqiang Index - Full Pipeline")
    print("=" * 70)

    total_start = time.time()
    results = []

    for i, (script, description) in enumerate(STEPS, 1):
        print(f"\n{'=' * 70}")
        print(f"Step {i}/{len(STEPS)}: {description}")
        print(f"Script: {script}")
        print("=" * 70)

        step_start = time.time()
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            cwd=str(SCRIPTS_DIR),
        )
        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\nFAILED: {script} exited with code {result.returncode}")
            print(f"Pipeline aborted at step {i}/{len(STEPS)}")
            sys.exit(1)

        results.append((script, elapsed))
        print(f"\nCompleted in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("Pipeline Complete")
    print("=" * 70)
    print(f"\nTotal time: {total_elapsed:.1f}s\n")
    print(f"{'Step':<35s} {'Time':>8s}")
    print("-" * 45)
    for script, elapsed in results:
        print(f"  {script:<33s} {elapsed:>6.1f}s")
    print("-" * 45)
    print(f"  {'TOTAL':<33s} {total_elapsed:>6.1f}s")
    print()


if __name__ == "__main__":
    main()
