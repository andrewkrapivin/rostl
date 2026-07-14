#!/usr/bin/env python3
"""Summarize cold-start fractional sweeps and render dependency-free SVGs."""

import csv
import glob
import math
import os
import re


ROOT = "analysis/lane_oram_security"
RUNS = os.path.join(ROOT, "cold_start_runs")
COLORS = {(3, 1): "#0072b2", (2, 2): "#d55e00", (4, 1): "#009e73"}


def identity(path):
    match = re.search(r"_z(\d+)_y(\d+)_r(\d+)_(\d+)_(?:tail|epochs)\.csv$", path)
    if not match:
        raise ValueError(f"unrecognized cold-start filename: {path}")
    return tuple(map(int, match.groups()))


def rows(path, metric):
    with open(path, newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["metric"] == metric]


def distribution(path, metric):
    data = rows(path, metric)
    mean = sum(float(row["probability"]) for row in data)
    minimum = next(int(row["threshold"]) for row in data if float(row["probability"]) < 1.0)
    maximum = int(data[-1]["threshold"])
    return mean, minimum, maximum


def observed_capacities(path, metric="insertion"):
    data = rows(path, metric)
    result = {}
    for exponent in (8, 16, 20, 24):
        target = 2.0**-exponent
        result[exponent] = next(
            int(row["threshold"]) for row in data if float(row["probability"]) <= target
        )
    return result


def epochs(path, metric="post"):
    data = []
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["metric"] == metric and row["threshold"] == "0":
                data.append(
                    {
                        "epoch": int(row["epoch"]),
                        "first": int(row["first_operation"]),
                        "last": int(row["last_operation"]),
                        "mean": float(row["mean"]),
                        "minimum": int(row["minimum"]),
                        "maximum": int(row["maximum"]),
                        "n": int(row["n"]),
                        "epoch_operations": int(row["epoch_operations"]),
                    }
                )
    return data


def window_summary(points):
    return sum(point["mean"] for point in points) / len(points), max(point["maximum"] for point in points)


def write_summary(records):
    path = os.path.join(ROOT, "fractional_cold_start_summary.csv")
    header = [
        "b", "z", "y", "rate_numerator", "rate_denominator", "n", "operations",
        "epoch_operations", "post_mean", "post_max", "insertion_mean", "insertion_max",
        "first_epoch_post_mean", "first_epoch_post_max", "first_cycle_post_mean",
        "first_cycle_post_max", "last_cycle_post_mean", "last_cycle_post_max",
        "first_to_last_cycle_mean_ratio", "insertion_capacity_2^-8",
        "insertion_capacity_2^-16", "insertion_capacity_2^-20",
        "insertion_capacity_2^-24",
    ]
    with open(path, "w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        for record in records:
            points = record["epochs"]
            epochs_per_cycle = points[0]["n"] // points[0]["epoch_operations"]
            first_mean, first_max = window_summary(points[:epochs_per_cycle])
            last_mean, last_max = window_summary(points[-epochs_per_cycle:])
            writer.writerow(
                [
                    2,
                    record["z"],
                    record["y"],
                    record["numerator"],
                    record["denominator"],
                    points[0]["n"],
                    points[-1]["last"],
                    points[0]["epoch_operations"],
                    f"{record['post'][0]:.9f}",
                    record["post"][2],
                    f"{record['insertion'][0]:.9f}",
                    record["insertion"][2],
                    f"{points[0]['mean']:.9f}",
                    points[0]["maximum"],
                    f"{first_mean:.9f}",
                    first_max,
                    f"{last_mean:.9f}",
                    last_max,
                    f"{first_mean / last_mean:.6f}",
                    record["capacities"][8],
                    record["capacities"][16],
                    record["capacities"][20],
                    record["capacities"][24],
                ]
            )
    return path


def figure3(records):
    width, height = 1220, 760
    left, top, plot_width, plot_height = 100, 90, 820, 590
    xmax = 25
    ymax = math.ceil(max(record["insertion"][2] for record in records) / 5) * 5
    x = lambda value: left + value / xmax * plot_width
    y = lambda value: top + (1 - value / ymax) * plot_height
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="610" y="31" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">Cold-start Lane ORAM insertion-demand tail</text>',
        '<text x="610" y="57" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#444">N = 2²⁰, 2²⁵ measured accesses immediately after randomized initialization; no discarded warm-up</text>',
    ]
    for tick in range(0, xmax + 1, 5):
        px = x(tick)
        out.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top + plot_height}" stroke="#e2e2e2"/>')
        out.append(f'<text x="{px}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="13">{tick}</text>')
    for tick in range(0, ymax + 1, 5):
        py = y(tick)
        out.append(f'<line x1="{left}" y1="{py}" x2="{left + plot_width}" y2="{py}" stroke="#e2e2e2"/>')
        out.append(f'<text x="{left - 12}" y="{py + 5}" text-anchor="end" font-family="sans-serif" font-size="13">{tick}</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<text x="{left + plot_width / 2}" y="738" text-anchor="middle" font-family="sans-serif" font-size="16">Security exponent k = log₂(1 / Pr[insertion demand &gt; S])</text>')
    out.append(f'<text x="25" y="{top + plot_height / 2}" transform="rotate(-90 25 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="16">Stash capacity S (blocks)</text>')
    for index, record in enumerate(records):
        color = COLORS[(record["z"], record["y"])]
        dash = "" if record["numerator"] == 1 else ' stroke-dasharray="9 5"'
        points = []
        for row in rows(record["tail"], "insertion"):
            probability = float(row["probability"])
            if 0 < probability < 0.9:
                exponent = -math.log2(probability)
                if exponent <= xmax:
                    points.append(f"{x(exponent):.2f},{y(int(row['threshold'])):.2f}")
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.8"{dash} points="{" ".join(points)}"/>')
        ly = 125 + index * 54
        out.append(f'<line x1="965" y1="{ly}" x2="1000" y2="{ly}" stroke="{color}" stroke-width="4"{dash}/>')
        out.append(f'<text x="1010" y="{ly + 5}" font-family="sans-serif" font-size="14">Z={record["z"]}, Y={record["y"]}, r={record["numerator"]}/{record["denominator"]}</text>')
        out.append(f'<text x="1010" y="{ly + 24}" font-family="sans-serif" font-size="12" fill="#555">mean/max demand {record["insertion"][0]:.3f}/{record["insertion"][2]}</text>')
    out.append('<rect x="950" y="86" width="245" height="355" fill="none" stroke="#cccccc" rx="5"/>')
    out.append('<text x="965" y="476" font-family="sans-serif" font-size="12" fill="#444">solid: r=1/2; dashed: r=2/3</text>')
    out.append('<text x="965" y="500" font-family="sans-serif" font-size="12" fill="#444">Direct resolution: 2⁻²⁵</text>')
    out.append('</svg>')
    path = os.path.join(ROOT, "fractional_cold_start_figure3.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(out))
    return path


def evolution(records):
    width, height = 1220, 820
    left, plot_width, panel_height = 100, 820, 270
    panel_tops = {(1, 2): 100, (2, 3): 475}
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="610" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">Cold-start stash evolution</text>',
        '<text x="610" y="58" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#444">Each point is the mean post-eviction stash over 2¹⁶ accesses; one key cycle is N accesses</text>',
    ]
    for rate in [(1, 2), (2, 3)]:
        selected = [record for record in records if (record["numerator"], record["denominator"]) == rate]
        top = panel_tops[rate]
        ymax = max(point["mean"] for record in selected for point in record["epochs"]) * 1.12
        ymax = max(1.0, math.ceil(ymax * 2) / 2)
        x = lambda value: left + value / 32 * plot_width
        y = lambda value: top + (1 - value / ymax) * panel_height
        for tick in range(0, 33, 4):
            px = x(tick)
            out.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top + panel_height}" stroke="#e2e2e2"/>')
            out.append(f'<text x="{px}" y="{top + panel_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick}</text>')
        for index in range(5):
            value = ymax * index / 4
            py = y(value)
            out.append(f'<line x1="{left}" y1="{py}" x2="{left + plot_width}" y2="{py}" stroke="#e2e2e2"/>')
            out.append(f'<text x="{left - 12}" y="{py + 5}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>')
        one_cycle = x(1)
        out.append(f'<line x1="{one_cycle}" y1="{top}" x2="{one_cycle}" y2="{top + panel_height}" stroke="#9a3d00" stroke-width="1.5" stroke-dasharray="6 4"/>')
        out.append(f'<text x="{one_cycle + 6}" y="{top + 18}" font-family="sans-serif" font-size="11" fill="#7b3300">end first key cycle</text>')
        out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" stroke="black"/>')
        out.append(f'<line x1="{left}" y1="{top + panel_height}" x2="{left + plot_width}" y2="{top + panel_height}" stroke="black"/>')
        out.append(f'<text x="{left}" y="{top - 12}" font-family="sans-serif" font-size="16" font-weight="600">r = {rate[0]}/{rate[1]}</text>')
        for record in selected:
            color = COLORS[(record["z"], record["y"])]
            points = " ".join(
                f'{x((point["first"] + point["last"]) / 2 / point["n"]):.2f},{y(point["mean"]):.2f}'
                for point in record["epochs"]
            )
            out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')
            epochs_per_cycle = record["epochs"][0]["n"] // record["epochs"][0]["epoch_operations"]
            first_mean, _ = window_summary(record["epochs"][:epochs_per_cycle])
            last_mean, _ = window_summary(record["epochs"][-epochs_per_cycle:])
            ly = top + 50 + selected.index(record) * 56
            out.append(f'<line x1="965" y1="{ly}" x2="1000" y2="{ly}" stroke="{color}" stroke-width="4"/>')
            out.append(f'<text x="1010" y="{ly + 5}" font-family="sans-serif" font-size="14">Z={record["z"]}, Y={record["y"]}</text>')
            out.append(f'<text x="1010" y="{ly + 24}" font-family="sans-serif" font-size="12" fill="#555">cycle 1 → 32: {first_mean:.3f} → {last_mean:.3f}</text>')
    out.append('<text x="510" y="806" text-anchor="middle" font-family="sans-serif" font-size="16">Completed key cycles after randomized initialization</text>')
    out.append('<text x="23" y="410" transform="rotate(-90 23 410)" text-anchor="middle" font-family="sans-serif" font-size="16">Mean post-eviction stash</text>')
    out.append('</svg>')
    path = os.path.join(ROOT, "fractional_cold_start_evolution.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(out))
    return path


def main():
    records = []
    for tail in sorted(glob.glob(os.path.join(RUNS, "cold_b2_*_tail.csv"))):
        z, y, numerator, denominator = identity(tail)
        epoch_path = tail.replace("_tail.csv", "_epochs.csv")
        records.append(
            {
                "z": z,
                "y": y,
                "numerator": numerator,
                "denominator": denominator,
                "tail": tail,
                "post": distribution(tail, "post"),
                "insertion": distribution(tail, "insertion"),
                "capacities": observed_capacities(tail),
                "epochs": epochs(epoch_path),
            }
        )
    if len(records) != 6:
        raise RuntimeError(f"expected six completed cold-start runs, found {len(records)}")
    records.sort(key=lambda record: (record["numerator"] / record["denominator"], record["z"] * record["y"], record["z"]))
    for path in [write_summary(records), figure3(records), evolution(records)]:
        print(path)


if __name__ == "__main__":
    main()
