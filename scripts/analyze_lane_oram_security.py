#!/usr/bin/env python3
"""Summarize Lane ORAM security CSVs and render dependency-free SVG plots."""

import csv
import glob
import json
import math
import os
import re


RESULTS = "analysis/lane_oram_security"
COLORS = {
    "B=2, Z=3": "#0072b2",
    "B=4, Z=5": "#d55e00",
    "B=4, Z=6": "#e69f00",
    "B=8, Z=9": "#009e73",
    "B=8, Z=10": "#cc79a7",
}


def read_rows(path, metric):
    with open(path, newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["metric"] == metric]


def distribution_summary(rows):
    probabilities = [float(row["probability"]) for row in rows]
    mean = sum(probabilities)
    minimum = next(int(row["threshold"]) for row in rows if float(row["probability"]) < 1.0)
    maximum = int(rows[-1]["threshold"])
    capacities = {}
    for exponent in (1, 4, 8, 12, 16, 20, 24):
        target = 2.0 ** -exponent
        capacities[exponent] = next(
            int(row["threshold"]) for row in rows if float(row["probability"]) <= target
        )
    return mean, minimum, maximum, capacities


def config_from_row(row):
    return f"B={row['b']}, Z={row['z']}"


def write_primary_summary():
    output_path = os.path.join(RESULTS, "primary_summary.csv")
    fields = [
        "configuration", "n", "warmup", "operations", "seed",
        "post_mean", "post_min", "post_max",
        "insertion_mean", "insertion_min", "insertion_max",
    ] + [f"capacity_for_2^-{value}" for value in (1, 4, 8, 12, 16, 20, 24)]
    with open(output_path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for path in sorted(glob.glob(os.path.join(RESULTS, "primary_b*_z*.csv"))):
            post = read_rows(path, "post")
            insertion = read_rows(path, "insertion")
            post_mean, post_min, post_max, _ = distribution_summary(post)
            insertion_mean, insertion_min, insertion_max, capacities = distribution_summary(insertion)
            first = insertion[0]
            row = {
                "configuration": config_from_row(first),
                "n": first["n"],
                "warmup": first["warmup"],
                "operations": first["operations"],
                "seed": first["seed"],
                "post_mean": f"{post_mean:.6f}",
                "post_min": post_min,
                "post_max": post_max,
                "insertion_mean": f"{insertion_mean:.6f}",
                "insertion_min": insertion_min,
                "insertion_max": insertion_max,
            }
            for exponent, capacity in capacities.items():
                row[f"capacity_for_2^-{exponent}"] = capacity
            writer.writerow(row)
    return output_path


def write_scaling_summary():
    output_path = os.path.join(RESULTS, "scaling_summary.csv")
    fields = ["configuration", "n", "post_mean", "stash_fraction", "post_min", "post_max"]
    with open(output_path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        records = []
        for path in glob.glob(os.path.join(RESULTS, "scaling_b*_z*_n*.csv")):
            rows = read_rows(path, "post")
            mean, minimum, maximum, _ = distribution_summary(rows)
            first = rows[0]
            n = int(first["n"])
            records.append((int(first["b"]), int(first["z"]), n, {
                "configuration": config_from_row(first),
                "n": n,
                "post_mean": f"{mean:.6f}",
                "stash_fraction": f"{mean / n:.9f}",
                "post_min": minimum,
                "post_max": maximum,
            }))
        for _, _, _, row in sorted(records):
            writer.writerow(row)
    return output_path


def write_replicate_summary():
    output_path = os.path.join(RESULTS, "replicate_summary.csv")
    fields = ["configuration", "seed", "operations", "post_mean", "insertion_mean"]
    records = []
    for path in glob.glob(os.path.join(RESULTS, "replicate_b*_z*_s*.csv")):
        post = read_rows(path, "post")
        insertion = read_rows(path, "insertion")
        post_mean, _, _, _ = distribution_summary(post)
        insertion_mean, _, _, _ = distribution_summary(insertion)
        first = post[0]
        records.append((int(first["b"]), int(first["z"]), int(first["seed"]), {
            "configuration": config_from_row(first),
            "seed": first["seed"],
            "operations": first["operations"],
            "post_mean": f"{post_mean:.6f}",
            "insertion_mean": f"{insertion_mean:.6f}",
        }))
    with open(output_path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for _, _, _, row in sorted(records):
            writer.writerow(row)
    return output_path


def write_workload_summary():
    output_path = os.path.join(RESULTS, "workload_summary.csv")
    fields = ["workload", "post_mean", "insertion_mean", "post_min", "post_max"]
    with open(output_path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for path in sorted(glob.glob(os.path.join(RESULTS, "workload_b2_z3_*.csv"))):
            post = read_rows(path, "post")
            insertion = read_rows(path, "insertion")
            post_mean, post_min, post_max, _ = distribution_summary(post)
            insertion_mean, _, _, _ = distribution_summary(insertion)
            writer.writerow({
                "workload": post[0]["workload"],
                "post_mean": f"{post_mean:.6f}",
                "insertion_mean": f"{insertion_mean:.6f}",
                "post_min": post_min,
                "post_max": post_max,
            })
    return output_path


def write_benchmark_summary():
    root = "target/criterion/ORAM_Random_Update_WallTime"
    pattern = os.path.join(root, "LaneORAM_56B_B*_Z*", "*", "new", "estimates.json")
    output_path = os.path.join(RESULTS, "benchmark_summary.csv")
    records = []
    for path in glob.glob(pattern):
        match = re.search(r"LaneORAM_56B_B(\d+)_Z(\d+)/(\d+)/new", path)
        if not match:
            continue
        b, z, log_n = map(int, match.groups())
        if (b, z) not in ((4, 6), (8, 9), (8, 10)):
            continue
        with open(path, encoding="utf-8") as source:
            slope = json.load(source)["slope"]
        records.append((b, z, log_n, slope))
    with open(output_path, "w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["configuration", "log_n", "nanoseconds", "ci_low", "ci_high"])
        for b, z, log_n, slope in sorted(records):
            interval = slope["confidence_interval"]
            writer.writerow([
                f"B={b}, Z={z}", log_n, f"{slope['point_estimate']:.6f}",
                f"{interval['lower_bound']:.6f}", f"{interval['upper_bound']:.6f}",
            ])
    return output_path


def svg_header(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">{title}</text>',
    ]


def render_figure3a():
    width, height = 1000, 700
    left, right, top, bottom = 95, 35, 65, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    x_max, y_max = 24.0, 3500.0
    x = lambda value: left + value / x_max * plot_width
    y = lambda value: top + (1.0 - value / y_max) * plot_height
    lines = svg_header(width, height, "Lane ORAM transient stash failure tail (N = 4096)")
    for tick in range(0, 25, 4):
        px = x(tick)
        lines.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top + plot_height}" stroke="#dddddd"/>')
        lines.append(f'<text x="{px}" y="{top + plot_height + 28}" text-anchor="middle" font-family="sans-serif" font-size="14">{tick}</text>')
    for tick in range(0, 3501, 500):
        py = y(tick)
        lines.append(f'<line x1="{left}" y1="{py}" x2="{left + plot_width}" y2="{py}" stroke="#dddddd"/>')
        lines.append(f'<text x="{left - 12}" y="{py + 5}" text-anchor="end" font-family="sans-serif" font-size="14">{tick}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>')
    lines.append(f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="17">log₂(1 / per-operation failure probability)</text>')
    lines.append(f'<text x="25" y="{top + plot_height / 2}" transform="rotate(-90 25 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="17">required stash capacity R</text>')

    for path in sorted(glob.glob(os.path.join(RESULTS, "primary_b*_z*.csv"))):
        rows = read_rows(path, "insertion")
        label = config_from_row(rows[0])
        points = []
        for row in rows:
            probability = float(row["probability"])
            if 0.0 < probability < 0.85:
                exponent = -math.log2(probability)
                if exponent <= x_max:
                    points.append(f"{x(exponent):.2f},{y(int(row['threshold'])):.2f}")
        lines.append(f'<polyline fill="none" stroke="{COLORS[label]}" stroke-width="3" points="{" ".join(points)}"/>')

    stash20_y = y(20)
    lines.append(f'<line x1="{left}" y1="{stash20_y}" x2="{left + plot_width}" y2="{stash20_y}" stroke="#bb0000" stroke-width="2" stroke-dasharray="8 6"/>')
    lines.append(f'<text x="{left + 8}" y="{stash20_y - 8}" font-family="sans-serif" font-size="13" fill="#990000">S=20 (failure observed on every measured operation)</text>')
    legend_x, legend_y = left + 25, top + 25
    for index, (label, color) in enumerate(COLORS.items()):
        py = legend_y + index * 25
        lines.append(f'<line x1="{legend_x}" y1="{py}" x2="{legend_x + 32}" y2="{py}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{legend_x + 42}" y="{py + 5}" font-family="sans-serif" font-size="14">{label}</text>')
    lines.append('</svg>')
    path = os.path.join(RESULTS, "figure3a_lane_oram.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(lines))
    return path


def render_scaling():
    width, height = 1000, 650
    left, right, top, bottom = 90, 35, 65, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max, y_max = 6.0, 15.0, 0.9
    x = lambda value: left + (value - x_min) / (x_max - x_min) * plot_width
    y = lambda value: top + (1.0 - value / y_max) * plot_height
    lines = svg_header(width, height, "Lane ORAM stash fraction grows with capacity")
    for tick in range(6, 16):
        px = x(tick)
        lines.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top + plot_height}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{px}" y="{top + plot_height + 27}" text-anchor="middle" font-family="sans-serif" font-size="14">{tick}</text>')
    for index in range(0, 10):
        value = index / 10
        py = y(value)
        lines.append(f'<line x1="{left}" y1="{py}" x2="{left + plot_width}" y2="{py}" stroke="#dddddd"/>')
        lines.append(f'<text x="{left - 12}" y="{py + 5}" text-anchor="end" font-family="sans-serif" font-size="14">{value:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>')
    lines.append(f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="17">log₂(N)</text>')
    lines.append(f'<text x="25" y="{top + plot_height / 2}" transform="rotate(-90 25 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="17">mean post-eviction stash / N</text>')

    records = {}
    with open(os.path.join(RESULTS, "scaling_summary.csv"), newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            records.setdefault(row["configuration"], []).append((math.log2(int(row["n"])), float(row["stash_fraction"])))
    for label, points in records.items():
        points.sort()
        encoded = " ".join(f"{x(px):.2f},{y(py):.2f}" for px, py in points)
        lines.append(f'<polyline fill="none" stroke="{COLORS[label]}" stroke-width="3" points="{encoded}"/>')
        for px, py in points:
            lines.append(f'<circle cx="{x(px):.2f}" cy="{y(py):.2f}" r="4" fill="{COLORS[label]}"/>')
    legend_x, legend_y = left + 25, top + 25
    for index, (label, color) in enumerate(COLORS.items()):
        py = legend_y + index * 25
        lines.append(f'<line x1="{legend_x}" y1="{py}" x2="{legend_x + 32}" y2="{py}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{legend_x + 42}" y="{py + 5}" font-family="sans-serif" font-size="14">{label}</text>')
    lines.append('</svg>')
    path = os.path.join(RESULTS, "stash_scaling.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(lines))
    return path


def main():
    outputs = [
        write_primary_summary(),
        write_scaling_summary(),
        write_replicate_summary(),
        write_workload_summary(),
        write_benchmark_summary(),
        render_figure3a(),
        render_scaling(),
    ]
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
