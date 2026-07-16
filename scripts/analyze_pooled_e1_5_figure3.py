#!/usr/bin/env python3
"""Analyze pooled Y=2, E=1.5 stash tails and render a Circuit-ORAM Figure-3 analogue."""

import csv
import glob
import math
import os
import re
import sys


CONTROL_RUNS = "analysis/lane_oram_security/pooled_e1_5_figure3_runs"
PRIMARY_RUNS = "analysis/lane_oram_security/pooled_e1_5_figure3_n20_runs"
OUT = "analysis/lane_oram_security"
LOADS = (100, 105, 110, 115, 120, 125)
WIDTHS = (8, 16)
TARGET_BITS = 96.0
COLORS = {
    100: "#0072b2",
    105: "#56b4e9",
    110: "#009e73",
    115: "#e69f00",
    120: "#d55e00",
    125: "#cc79a7",
}


def read_metric(path, metric):
    with open(path, newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["metric"] == metric]


def linear_fit(points):
    if len(points) < 4:
        return None
    n = len(points)
    x_mean = sum(x for x, _ in points) / n
    y_mean = sum(y for _, y in points) / n
    sxx = sum((x - x_mean) ** 2 for x, _ in points)
    if sxx == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / sxx
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in points]
    rss = sum(value * value for value in residuals)
    residual = math.sqrt(rss / (n - 2))
    total = sum((y - y_mean) ** 2 for _, y in points)
    r_squared = 1.0 - rss / total if total else 1.0
    prediction = intercept + slope * TARGET_BITS
    mean_prediction_se = residual * math.sqrt(
        1.0 / n + (TARGET_BITS - x_mean) ** 2 / sxx
    )
    return {
        "intercept": intercept,
        "slope": slope,
        "residual": residual,
        "r_squared": r_squared,
        "prediction": prediction,
        "prediction_se": mean_prediction_se,
        "points": n,
        "min_bits": min(x for x, _ in points),
        "max_bits": max(x for x, _ in points),
    }


def fit_tail(rows, minimum_bits=8.0, minimum_count=64):
    points = []
    for row in rows:
        count = int(row["exceed_count"])
        probability = float(row["probability"])
        if count < minimum_count or probability <= 0.0:
            continue
        bits = -math.log2(probability)
        if bits >= minimum_bits:
            points.append((bits, float(row["threshold"])))
    return linear_fit(points)


def sensitivity_predictions(rows):
    estimates = []
    diagnostics = []
    for minimum_bits in (6.0, 8.0, 10.0, 12.0):
        for minimum_count in (32, 64, 256, 1024):
            fit = fit_tail(rows, minimum_bits, minimum_count)
            if fit is None:
                continue
            estimates.append(fit["prediction"])
            diagnostics.append((minimum_bits, minimum_count, fit))
    return estimates, diagnostics


def epoch_predictions(path, metric):
    grouped = {}
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["metric"] == metric:
                grouped.setdefault(int(row["epoch"]), []).append(row)
    estimates = []
    fits = []
    for epoch, rows in sorted(grouped.items()):
        fit = fit_tail(rows, minimum_bits=6.0, minimum_count=32)
        if fit is not None:
            estimates.append(fit["prediction"])
            fits.append((epoch, fit))
    return estimates, fits


def percentile(values, probability):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def ceil_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def control_path(width, load, suffix=""):
    base = os.path.join(CONTROL_RUNS, f"m10_x{width}_rho{load}_e1_5_ops27")
    return base + suffix + ".csv"


def primary_path(width, load, suffix=""):
    capacity = 1 << 20
    leaves = capacity // width
    base = os.path.join(
        PRIMARY_RUNS, f"cap20_x{width}_m{leaves}_rho{load}_e1_5_ops25"
    )
    return base + suffix + ".csv"


def measured_points(rows):
    result = []
    for row in rows:
        probability = float(row["probability"])
        if probability <= 0.0:
            continue
        bits = -math.log2(probability)
        if 1.0 <= bits <= 32.0:
            result.append((bits, float(row["threshold"]), int(row["exceed_count"])))
    return result


def analyze_metric(path, epoch_path, metric, dataset, width, load, diagnostics_rows):
    rows = read_metric(path, metric)
    fit = fit_tail(rows)
    if fit is None:
        raise RuntimeError(f"insufficient tail points in {path} ({metric})")
    sensitivity, window_fits = sensitivity_predictions(rows)
    epochs, epoch_fits = epoch_predictions(epoch_path, metric)
    result = {
        "rows": rows,
        "measured": measured_points(rows),
        "fit": fit,
        "observed_max": int(rows[-1]["threshold"]),
        "sensitivity_low": min(sensitivity),
        "sensitivity_high": max(sensitivity),
        "epoch_low": min(epochs) if epochs else math.nan,
        "epoch_high": max(epochs) if epochs else math.nan,
        "epoch_p95": percentile(epochs, 0.95),
        "upper_line": fit["prediction"] + 1.96 * fit["prediction_se"],
    }
    for minimum_bits, minimum_count, window_fit in window_fits:
        diagnostics_rows.append({
            "dataset": dataset,
            "x": width,
            "load_percent": load,
            "metric": metric,
            "source": "full-window",
            "minimum_bits": minimum_bits,
            "minimum_count": minimum_count,
            "epoch": "",
            "points": window_fit["points"],
            "fit_min_bits": window_fit["min_bits"],
            "fit_max_bits": window_fit["max_bits"],
            "intercept": window_fit["intercept"],
            "slope_blocks_per_bit": window_fit["slope"],
            "residual_blocks": window_fit["residual"],
            "r_squared": window_fit["r_squared"],
            "estimate_at_2^-96": window_fit["prediction"],
        })
    for epoch, epoch_fit in epoch_fits:
        diagnostics_rows.append({
            "dataset": dataset,
            "x": width,
            "load_percent": load,
            "metric": metric,
            "source": "epoch",
            "minimum_bits": 6,
            "minimum_count": 32,
            "epoch": epoch,
            "points": epoch_fit["points"],
            "fit_min_bits": epoch_fit["min_bits"],
            "fit_max_bits": epoch_fit["max_bits"],
            "intercept": epoch_fit["intercept"],
            "slope_blocks_per_bit": epoch_fit["slope"],
            "residual_blocks": epoch_fit["residual"],
            "r_squared": epoch_fit["r_squared"],
            "estimate_at_2^-96": epoch_fit["prediction"],
        })
    return result


def analyze():
    records = {}
    diagnostics_rows = []
    recommendation_rows = []
    for width in WIDTHS:
        for load in LOADS:
            paths = {
                "primary": (primary_path(width, load), primary_path(width, load, "_epochs")),
                "control": (control_path(width, load), control_path(width, load, "_epochs")),
            }
            for path, epoch_path in paths.values():
                if not os.path.exists(path) or not os.path.exists(epoch_path):
                    raise FileNotFoundError(
                        f"missing result for X={width}, rho={load}: {path}"
                    )
            datasets = {}
            for dataset, (path, epoch_path) in paths.items():
                datasets[dataset] = {
                    metric: analyze_metric(
                        path, epoch_path, metric, dataset, width, load, diagnostics_rows
                    )
                    for metric in ("post", "insertion")
                }
            primary = datasets["primary"]
            control = datasets["control"]
            records[(width, load)] = primary
            insertion = primary["insertion"]
            recommendation_rows.append({
                "x": width,
                "load_percent": load,
                "logical_blocks": (1 << 20) * load // 100,
                "primary_samples": 1 << 25,
                "control_samples": 1 << 27,
                "post_observed_max": primary["post"]["observed_max"],
                "insertion_observed_max": insertion["observed_max"],
                "post_fit_intercept": primary["post"]["fit"]["intercept"],
                "post_slope_blocks_per_bit": primary["post"]["fit"]["slope"],
                "post_r_squared": primary["post"]["fit"]["r_squared"],
                "post_point_estimate_2^-96": primary["post"]["fit"]["prediction"],
                "insertion_point_estimate_2^-96": insertion["fit"]["prediction"],
                "insertion_fit_95_upper": insertion["upper_line"],
                "insertion_sensitivity_low": insertion["sensitivity_low"],
                "insertion_sensitivity_high": insertion["sensitivity_high"],
                "insertion_epoch_low": insertion["epoch_low"],
                "insertion_epoch_high": insertion["epoch_high"],
                "insertion_epoch_p95": insertion["epoch_p95"],
                "control_insertion_point_estimate_2^-96": control["insertion"]["fit"]["prediction"],
                "control_insertion_sensitivity_high": control["insertion"]["sensitivity_high"],
                "primary_minus_control_estimate": (
                    insertion["fit"]["prediction"] - control["insertion"]["fit"]["prediction"]
                ),
                "cross_size_sensitivity_high": max(
                    insertion["sensitivity_high"],
                    control["insertion"]["sensitivity_high"],
                ),
                "cross_size_and_epoch_upper": max(
                    insertion["epoch_p95"],
                    insertion["sensitivity_high"],
                    control["insertion"]["sensitivity_high"],
                ),
                "monotone_empirical_upper": 0,
                "engineering_recommendation": 0,
            })
    for width in WIDTHS:
        envelope = 0.0
        for row in recommendation_rows:
            if row["x"] != width:
                continue
            envelope = max(envelope, row["cross_size_and_epoch_upper"])
            row["monotone_empirical_upper"] = envelope
            row["engineering_recommendation"] = ceil_multiple(envelope + 8.0, 8)
    return records, diagnostics_rows, recommendation_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key, value in row.items():
                encoded[key] = f"{value:.9f}" if isinstance(value, float) else value
            writer.writerow(encoded)


def svg_text(x, y, value, **attrs):
    attributes = {
        "x": x,
        "y": y,
        "font-family": "sans-serif",
        **attrs,
    }
    encoded = " ".join(f'{key.replace("_", "-")}="{item}"' for key, item in attributes.items())
    return f"<text {encoded}>{value}</text>"


def render_figure(records):
    canvas_width, canvas_height = 1540, 850
    panel_top, panel_height = 130, 600
    panel_width = 620
    panel_lefts = {8: 100, 16: 835}
    predictions = [
        records[(width, load)]["post"]["fit"]["prediction"]
        for width in WIDTHS for load in LOADS
    ]
    y_max = max(40, int(math.ceil((max(predictions) + 8) / 20.0) * 20))
    x_max = 100.0

    def sx(left, value):
        return left + value / x_max * panel_width

    def sy(value):
        return panel_top + (1.0 - value / y_max) * panel_height

    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(canvas_width / 2, 34, "Pooled Circuit-pair stash tail by leaf-slot occupancy", **{
            "text-anchor": "middle", "font-size": 23, "font-weight": 600
        }),
        svg_text(canvas_width / 2, 62, "Y=2, E=1.5, total bottom capacity XM=2²⁰; 2²³ warm-up + 2²⁵ measured cyclic accesses", **{
            "text-anchor": "middle", "font-size": 15, "fill": "#333"
        }),
        svg_text(canvas_width / 2, 84, "Solid: empirical P[post-eviction stash > R] · dashed: log-linear fit extrapolated to 2⁻⁹⁶", **{
            "text-anchor": "middle", "font-size": 13, "fill": "#555"
        }),
    ]
    y_step = 20 if y_max <= 160 else 40
    for width in WIDTHS:
        left = panel_lefts[width]
        leaves_exponent = 20 - int(math.log2(width))
        for tick in range(0, 101, 16):
            px = sx(left, tick)
            output.append(f'<line x1="{px:.2f}" y1="{panel_top}" x2="{px:.2f}" y2="{panel_top + panel_height}" stroke="#e5e5e5"/>')
            output.append(svg_text(px, panel_top + panel_height + 24, tick, **{
                "text-anchor": "middle", "font-size": 12, "fill": "#333"
            }))
        for tick in range(0, y_max + 1, y_step):
            py = sy(tick)
            output.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + panel_width}" y2="{py:.2f}" stroke="#e5e5e5"/>')
            output.append(svg_text(left - 12, py + 4, tick, **{
                "text-anchor": "end", "font-size": 12, "fill": "#333"
            }))
        target_x = sx(left, 96)
        output.append(f'<line x1="{target_x:.2f}" y1="{panel_top}" x2="{target_x:.2f}" y2="{panel_top + panel_height}" stroke="#8b1a1a" stroke-width="1.8" stroke-dasharray="7 5"/>')
        output.append(svg_text(target_x - 5, panel_top + 18, "2⁻⁹⁶", **{
            "text-anchor": "end", "font-size": 12, "fill": "#8b1a1a", "font-weight": 600
        }))
        output.append(f'<line x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_top + panel_height}" stroke="#111" stroke-width="1.5"/>')
        output.append(f'<line x1="{left}" y1="{panel_top + panel_height}" x2="{left + panel_width}" y2="{panel_top + panel_height}" stroke="#111" stroke-width="1.5"/>')
        output.append(svg_text(left + panel_width / 2, 113, f"Total width X={width}, leaves/ORAM M=2^{leaves_exponent}", **{
            "text-anchor": "middle", "font-size": 17, "font-weight": 600
        }))
        legend_x = left + 17
        legend_y = panel_top + 25
        for index, load in enumerate(LOADS):
            color = COLORS[load]
            y_legend = legend_y + index * 23
            estimate = math.ceil(records[(width, load)]["post"]["fit"]["prediction"])
            output.append(f'<line x1="{legend_x}" y1="{y_legend}" x2="{legend_x + 28}" y2="{y_legend}" stroke="{color}" stroke-width="3"/>')
            output.append(svg_text(legend_x + 36, y_legend + 4, f"ρ={load}%  (fit R₉₆≈{estimate})", **{
                "font-size": 12, "fill": "#222"
            }))
        for load in LOADS:
            color = COLORS[load]
            metric = records[(width, load)]["post"]
            points = [(bits, stash) for bits, stash, _ in metric["measured"] if bits <= 30.0]
            if points:
                encoded = " ".join(f"{sx(left, bits):.2f},{sy(stash):.2f}" for bits, stash in points)
                output.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.7" points="{encoded}"/>')
                for bits, stash in points:
                    output.append(f'<circle cx="{sx(left, bits):.2f}" cy="{sy(stash):.2f}" r="1.8" fill="{color}"/>')
            fit = metric["fit"]
            start = max(8.0, fit["min_bits"])
            output.append(
                f'<line x1="{sx(left, start):.2f}" y1="{sy(fit["intercept"] + fit["slope"] * start):.2f}" '
                f'x2="{sx(left, 96):.2f}" y2="{sy(fit["prediction"]):.2f}" '
                f'stroke="{color}" stroke-width="2" stroke-dasharray="7 5"/>'
            )
            output.append(f'<circle cx="{sx(left, 96):.2f}" cy="{sy(fit["prediction"]):.2f}" r="3.7" fill="white" stroke="{color}" stroke-width="2"/>')
    output.append(svg_text(canvas_width / 2, 820, "log₂(1 / per-operation exceedance probability)", **{
        "text-anchor": "middle", "font-size": 16
    }))
    output.append(svg_text(24, panel_top + panel_height / 2, "Post-eviction stash threshold R (blocks)", **{
        "text-anchor": "middle", "font-size": 16,
        "transform": f"rotate(-90 24 {panel_top + panel_height / 2})"
    }))
    output.append("</svg>")
    path = os.path.join(OUT, "pooled_e1_5_figure3.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(output))
    return path


def main():
    records, diagnostics, recommendations = analyze()
    diagnostics_path = os.path.join(OUT, "pooled_e1_5_tail_fit_diagnostics.csv")
    recommendations_path = os.path.join(OUT, "pooled_e1_5_stash_recommendations.csv")
    write_csv(diagnostics_path, diagnostics)
    write_csv(recommendations_path, recommendations)
    figure_path = render_figure(records)
    print(figure_path)
    print(recommendations_path)
    print(diagnostics_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
