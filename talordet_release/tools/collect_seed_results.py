#!/usr/bin/env python3
"""Select the best validation epoch and summarize multiple seed runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


MAP50 = "metrics/mAP50(B)"
MAP = "metrics/mAP50-95(B)"


def best_row(run: Path) -> dict[str, str]:
    path = run / "results.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"No result rows: {path}")
    return max(rows, key=lambda row: 0.1 * float(row[MAP50]) + 0.9 * float(row[MAP]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    results = []
    for run in args.runs:
        row = best_row(run)
        results.append((run.name, int(float(row["epoch"])), float(row[MAP50]), float(row[MAP])))

    print("run\tbest_epoch\tmAP50\tmAP50-95")
    for name, epoch, map50, map_value in results:
        print(f"{name}\t{epoch}\t{map50:.5f}\t{map_value:.5f}")
    map50_values, map_values = [r[2] for r in results], [r[3] for r in results]
    map50_sd = statistics.stdev(map50_values) if len(results) > 1 else 0.0
    map_sd = statistics.stdev(map_values) if len(results) > 1 else 0.0
    print(
        "mean+/-sample_sd\t-\t"
        f"{statistics.mean(map50_values):.5f}+/-{map50_sd:.5f}\t"
        f"{statistics.mean(map_values):.5f}+/-{map_sd:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
