#!/usr/bin/env python3
"""Combine four fixed-gap overnight replicates into one tail per load."""

import csv
import glob
import json
import math
import os
import re
import statistics


BASE = "analysis/lane_oram_security/fixed_gap5_overnight_runs"
LOADS = (110, 120)
REPLICATES = 4
EXPECTED_OPERATIONS = 1 << 30
EXPECTED_WARMUP = 1 << 24
METRICS = ("post", "insertion", "pre_batch")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def metric_rows(rows, metric):
    return [row for row in rows if row["metric"] == metric]


def survival(rows):
    return {
        int(row["threshold"]): int(row["exceed_count"]) / int(row["samples"])
        for row in rows
    }


def mean_from_tail(rows):
    return sum(survival(rows).values())


def aggregate_metric(runs, metric):
    selected = [metric_rows(rows, metric) for rows in runs]
    assert all(rows for rows in selected)
    samples = sum(int(rows[0]["samples"]) for rows in selected)
    maximum = max(max(int(row["threshold"]) for row in rows) for rows in selected)
    maps = [
        {int(row["threshold"]): int(row["exceed_count"]) for row in rows}
        for rows in selected
    ]
    result = []
    for threshold in range(maximum + 1):
        count = sum(values.get(threshold, 0) for values in maps)
        probability = count / samples
        bits = math.inf if count == 0 else -math.log2(probability)
        result.append(
            {
                "metric": metric,
                "samples": samples,
                "threshold": threshold,
                "exceed_count": count,
                "probability": probability,
                "log2_inverse": bits,
            }
        )
    return result


def linear_slope(rows, low, high):
    points = [
        (float(row["log2_inverse"]), float(row["threshold"]))
        for row in rows
        if int(row["exceed_count"]) >= 64
        and math.isfinite(float(row["log2_inverse"]))
        and low <= float(row["log2_inverse"]) < high
    ]
    if len(points) < 3:
        return math.nan
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return math.nan
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def ks_distance(left, right):
    left_tail = survival(left)
    right_tail = survival(right)
    maximum = max(max(left_tail), max(right_tail))
    return max(
        abs(left_tail.get(threshold, 0.0) - right_tail.get(threshold, 0.0))
        for threshold in range(maximum + 1)
    )


def log_seconds(path):
    with open(path, encoding="utf-8") as source:
        match = re.search(r"seconds=([0-9.]+)", source.read())
    return float(match.group(1)) if match else math.nan


def encode(value):
    if isinstance(value, float):
        return "inf" if math.isinf(value) else f"{value:.12g}"
    return value


def write_csv(path, rows):
    assert rows
    with open(path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: encode(value) for key, value in row.items()})


def main():
    os.makedirs(BASE, exist_ok=True)
    combined_summary = []
    replicate_summary = []
    agreement_summary = []
    manifest = {
        "gap": 5,
        "x": 16,
        "z": 8,
        "y": 2,
        "warmup_per_replicate": EXPECTED_WARMUP,
        "operations_per_replicate": EXPECTED_OPERATIONS,
        "replicates_per_load": REPLICATES,
        "total_operations_per_load": EXPECTED_OPERATIONS * REPLICATES,
        "loads": {},
    }

    for load in LOADS:
        paths = [
            os.path.join(BASE, f"rho{load}_rep{rep}_ops30.csv")
            for rep in range(REPLICATES)
        ]
        assert all(os.path.exists(path) for path in paths), paths
        runs = [read_rows(path) for path in paths]
        for rows in runs:
            first = rows[0]
            assert int(first["operations"]) == EXPECTED_OPERATIONS
            assert int(first["warmup"]) == EXPECTED_WARMUP
            assert int(first["n"]) == 65536
            assert int(first["z"]) == 8 and int(first["y"]) == 2
            assert int(first["deterministic_numerator"]) == 1
            assert int(first["deterministic_denominator"]) == 5

        combined_rows = []
        combined_by_metric = {}
        for metric in METRICS:
            aggregate = aggregate_metric(runs, metric)
            combined_by_metric[metric] = aggregate
            combined_rows.extend(
                {
                    "load_percent": load,
                    "gap": 5,
                    "replicates": REPLICATES,
                    "operations_per_replicate": EXPECTED_OPERATIONS,
                    "total_operations": EXPECTED_OPERATIONS * REPLICATES,
                    **row,
                }
                for row in aggregate
            )
            summary = {
                "load_percent": load,
                "metric": metric,
                "samples": int(aggregate[0]["samples"]),
                "mean": sum(float(row["probability"]) for row in aggregate),
                "maximum": max(int(row["threshold"]) for row in aggregate),
            }
            for low, high in ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28)):
                summary[f"slope_bits_{low}_{high}"] = linear_slope(aggregate, low, high)
            combined_summary.append(summary)

        combined_path = os.path.join(BASE, f"combined_rho{load}_ops32.csv")
        write_csv(combined_path, combined_rows)

        epoch_rows = []
        for rep in range(REPLICATES):
            epoch_path = os.path.join(BASE, f"rho{load}_rep{rep}_ops30_epochs.csv")
            rows = read_rows(epoch_path)
            assert rows
            for row in rows:
                epoch_rows.append({"replicate": rep, **row})
        write_csv(os.path.join(BASE, f"combined_rho{load}_epochs.csv"), epoch_rows)

        for rep, rows in enumerate(runs):
            for metric in METRICS:
                selected = metric_rows(rows, metric)
                replicate_summary.append(
                    {
                        "load_percent": load,
                        "replicate": rep,
                        "metric": metric,
                        "samples": int(selected[0]["samples"]),
                        "mean": mean_from_tail(selected),
                        "maximum": max(int(row["threshold"]) for row in selected),
                        "ks_vs_combined": ks_distance(selected, combined_by_metric[metric]),
                        "seconds": log_seconds(
                            os.path.join(BASE, f"rho{load}_rep{rep}_ops30.log")
                        ),
                    }
                )

        for metric in METRICS:
            selected = [row for row in replicate_summary if row["load_percent"] == load and row["metric"] == metric]
            means = [row["mean"] for row in selected]
            agreement_summary.append(
                {
                    "load_percent": load,
                    "metric": metric,
                    "mean_min": min(means),
                    "mean_max": max(means),
                    "mean_sd": statistics.pstdev(means),
                    "maximum_min": min(row["maximum"] for row in selected),
                    "maximum_max": max(row["maximum"] for row in selected),
                    "ks_vs_combined_max": max(row["ks_vs_combined"] for row in selected),
                }
            )

        manifest["loads"][str(load)] = {
            "partials": paths,
            "combined": combined_path,
            "epoch_file": os.path.join(BASE, f"combined_rho{load}_epochs.csv"),
        }

    write_csv(os.path.join(BASE, "combined_summary.csv"), combined_summary)
    write_csv(os.path.join(BASE, "replicate_summary.csv"), replicate_summary)
    write_csv(os.path.join(BASE, "replicate_agreement.csv"), agreement_summary)
    with open(os.path.join(BASE, "manifest.json"), "w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2)
        target.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
