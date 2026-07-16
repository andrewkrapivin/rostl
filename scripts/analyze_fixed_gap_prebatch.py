#!/usr/bin/env python3
"""Analyze fixed-gap pre-batch stash tails and long-run/ensemble agreement."""

import csv
import glob
import html
import math
import os
import statistics


BASE = "analysis/lane_oram_security/fixed_gap_prebatch_runs"
OUT = "analysis/lane_oram_security"
LOADS = (100, 105, 110, 115, 120, 125)
GAPS = (5, 6)
COLORS = {
    100: "#0072b2",
    105: "#56b4e9",
    110: "#009e73",
    115: "#e69f00",
    120: "#d55e00",
    125: "#cc79a7",
}


def read_metric(path, metric="pre_batch"):
    with open(path, newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["metric"] == metric]


def survival(rows):
    return {
        int(row["threshold"]): int(row["exceed_count"]) / int(row["samples"])
        for row in rows
    }


def sample_count(rows):
    return int(rows[0]["samples"])


def mean_from_survival(values):
    return sum(values.values())


def maximum_from_rows(rows):
    return max(int(row["threshold"]) for row in rows)


def tail_points(rows):
    result = []
    for row in rows:
        probability = float(row["probability"])
        count = int(row["exceed_count"])
        if probability <= 0.0 or count == int(row["samples"]):
            continue
        result.append((-math.log2(probability), float(row["threshold"]), count))
    return result


def linear_slope(points):
    if len(points) < 3:
        return math.nan
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return math.nan
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def local_slopes(rows):
    points = [(x, y) for x, y, count in tail_points(rows) if count >= 64]
    result = {}
    for low, high in ((4, 8), (8, 12), (12, 16)):
        result[f"slope_bits_{low}_{high}"] = linear_slope(
            [(x, y) for x, y in points if low <= x < high]
        )
    return result


def aggregate_runs(paths):
    runs = [read_metric(path) for path in paths]
    total = sum(sample_count(rows) for rows in runs)
    maximum = max(maximum_from_rows(rows) for rows in runs)
    counts = {}
    for threshold in range(maximum + 1):
        counts[threshold] = sum(
            next(
                (
                    int(row["exceed_count"])
                    for row in rows
                    if int(row["threshold"]) == threshold
                ),
                0,
            )
            for rows in runs
        )
    return total, counts


def points_from_counts(samples, counts):
    return [
        (-math.log2(count / samples), float(threshold), count)
        for threshold, count in sorted(counts.items())
        if 0 < count < samples
    ]


def epoch_means(path):
    rows = read_metric(path)
    return [
        float(row["mean"])
        for row in rows
        if int(row["threshold"]) == 0
    ]


def run_mean(path):
    return mean_from_survival(survival(read_metric(path)))


def svg_text(x, y, value, **attrs):
    defaults = {"x": x, "y": y, "font-family": "sans-serif"}
    defaults.update(attrs)
    encoded = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(item))}"'
        for key, item in defaults.items()
    )
    return f"<text {encoded}>{html.escape(str(value))}</text>"


def polyline(points, color, dash=None, width=2.4, opacity=1.0):
    if len(points) < 2:
        return ""
    attrs = (
        f'fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"'
    )
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline {attrs} points="{coords}"/>'


def render_gap(gap, records):
    width, height = 1320, 850
    left, right, top, bottom = 105, 45, 115, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max = 22.0
    observed_max = max(maximum_from_rows(records[load]) for load in LOADS)
    step = 5 if observed_max <= 50 else 10
    y_max = int(math.ceil((observed_max + 3) / step) * step)

    def sx(value):
        return left + value / x_max * plot_w

    def sy(value):
        return top + plot_h - value / y_max * plot_h

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            45,
            f"Fixed gap {gap}: pre-batch stash tail",
            text_anchor="middle",
            font_size="25",
            font_weight="bold",
        ),
        svg_text(
            width / 2,
            76,
            "X=16, Y=2, total bottom capacity 2²⁰; 2²¹ warm-up + 2²⁴ measured accesses",
            text_anchor="middle",
            font_size="15",
            fill="#444",
        ),
    ]

    for value in range(0, int(x_max) + 1, 2):
        x = sx(value)
        items.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e8e8e8"/>'
        )
        items.append(svg_text(x, top + plot_h + 28, value, text_anchor="middle", font_size="13"))

    for value in range(0, y_max + 1, step):
        y = sy(value)
        items.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e8e8e8"/>'
        )
        items.append(svg_text(left - 13, y + 5, value, text_anchor="end", font_size="13"))

    items.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.5"/>',
            svg_text(
                left + plot_w / 2,
                height - 30,
                "log₂(1 / Pr[pre-batch stash > R])",
                text_anchor="middle",
                font_size="18",
            ),
            svg_text(
                28,
                top + plot_h / 2,
                "Pre-batch stash threshold R (blocks)",
                text_anchor="middle",
                font_size="18",
                transform=f"rotate(-90 28 {top + plot_h / 2})",
            ),
        ]
    )

    robust_bits = None
    for load in LOADS:
        rows = records[load]
        points = [(x, y, count) for x, y, count in tail_points(rows) if x <= x_max]
        robust = [(sx(x), sy(y)) for x, y, count in points if count >= 64]
        sparse = [(sx(x), sy(y)) for x, y, count in points if count < 64]
        if robust and sparse:
            sparse.insert(0, robust[-1])
        items.append(polyline(robust, COLORS[load]))
        items.append(polyline(sparse, COLORS[load], dash="7 6", opacity=0.72))
        robust_bits = math.log2(sample_count(rows) / 64)

    if robust_bits is not None:
        x = sx(robust_bits)
        items.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#555" stroke-width="1.3" stroke-dasharray="3 5"/>'
        )
        items.append(
            svg_text(
                x - 6,
                top + 18,
                "64 exceedances",
                text_anchor="end",
                font_size="12",
                fill="#444",
            )
        )

    legend_x, legend_y = left + 18, top + 30
    for index, load in enumerate(LOADS):
        y = legend_y + index * 25
        items.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{COLORS[load]}" stroke-width="3"/>'
        )
        items.append(svg_text(legend_x + 43, y + 5, f"rho={load}%", font_size="14"))

    items.append(
        svg_text(
            left + plot_w - 12,
            top + plot_h - 14,
            "Solid: ≥64 exceedances; dashed: 1–63 exceedances",
            text_anchor="end",
            font_size="12",
            fill="#555",
        )
    )
    items.append("</svg>")
    path = os.path.join(OUT, f"fixed_gap{gap}_prebatch_figure3.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(items))
    return path


def render_by_load(records):
    width, height = 1600, 1080
    outer_x, outer_y = 75, 105
    gap_x, gap_y = 55, 85
    panel_w = (width - 2 * outer_x - 2 * gap_x) / 3
    panel_h = (height - 2 * outer_y - gap_y) / 2
    x_max = 22.0
    gap_colors = {5: "#0072b2", 6: "#d55e00"}
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            42,
            "Pre-batch stash tails by load factor",
            text_anchor="middle",
            font_size="25",
            font_weight="bold",
        ),
        svg_text(
            width / 2,
            72,
            "Fixed gaps 5 and 6; X=16, Y=2, total bottom capacity 2²⁰",
            text_anchor="middle",
            font_size="15",
            fill="#444",
        ),
    ]

    for index, load in enumerate(LOADS):
        column, row = index % 3, index // 3
        left = outer_x + column * (panel_w + gap_x)
        top = outer_y + row * (panel_h + gap_y)
        observed = max(maximum_from_rows(records[gap][load]) for gap in GAPS)
        step = 5 if observed <= 50 else 10
        y_max = int(math.ceil((observed + 3) / step) * step)

        def sx(value):
            return left + value / x_max * panel_w

        def sy(value):
            return top + panel_h - value / y_max * panel_h

        for value in range(0, int(x_max) + 1, 4):
            x = sx(value)
            items.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + panel_h}" stroke="#ededed"/>'
            )
            items.append(svg_text(x, top + panel_h + 21, value, text_anchor="middle", font_size="10"))
        for value in range(0, y_max + 1, step):
            y = sy(value)
            items.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_w}" y2="{y:.2f}" stroke="#ededed"/>'
            )
            items.append(svg_text(left - 8, y + 4, value, text_anchor="end", font_size="10"))
        items.append(
            f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#222"/>'
        )
        items.append(
            svg_text(
                left + panel_w / 2,
                top - 14,
                f"rho={load}%",
                text_anchor="middle",
                font_size="17",
                font_weight="bold",
            )
        )

        for gap in GAPS:
            points = [
                (x, y, count)
                for x, y, count in tail_points(records[gap][load])
                if x <= x_max
            ]
            robust = [(sx(x), sy(y)) for x, y, count in points if count >= 64]
            sparse = [(sx(x), sy(y)) for x, y, count in points if count < 64]
            if robust and sparse:
                sparse.insert(0, robust[-1])
            items.append(polyline(robust, gap_colors[gap], width=2.2))
            items.append(polyline(sparse, gap_colors[gap], dash="6 5", width=2.2, opacity=0.72))

    items.append(
        svg_text(
            width / 2,
            height - 48,
            "log₂(1 / Pr[pre-batch stash > R])",
            text_anchor="middle",
            font_size="17",
        )
    )
    items.append(
        svg_text(
            24,
            height / 2,
            "Pre-batch stash threshold R (blocks)",
            text_anchor="middle",
            font_size="17",
            transform=f"rotate(-90 24 {height / 2})",
        )
    )
    legend_y = height - 19
    items.append(
        f'<line x1="{width/2-150}" y1="{legend_y}" x2="{width/2-112}" y2="{legend_y}" stroke="{gap_colors[5]}" stroke-width="3"/>'
    )
    items.append(svg_text(width / 2 - 102, legend_y + 5, "gap 5", font_size="13"))
    items.append(
        f'<line x1="{width/2+30}" y1="{legend_y}" x2="{width/2+68}" y2="{legend_y}" stroke="{gap_colors[6]}" stroke-width="3"/>'
    )
    items.append(svg_text(width / 2 + 78, legend_y + 5, "gap 6", font_size="13"))
    items.append("</svg>")
    path = os.path.join(OUT, "fixed_gap_prebatch_by_load.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(items))
    return path


def compare_long_and_ensemble(gap, load):
    long_rows = read_metric(
        os.path.join(BASE, f"fixedgap{gap}_x16_rho{load}_ops24.csv")
    )
    rep_paths = sorted(
        glob.glob(os.path.join(BASE, f"ensemble_gap{gap}_x16_rho{load}_rep*_ops20.csv"))
    )
    ensemble_samples, ensemble_counts = aggregate_runs(rep_paths)
    long_values = survival(long_rows)
    ensemble_values = {
        threshold: count / ensemble_samples
        for threshold, count in ensemble_counts.items()
    }
    maximum = max(max(long_values), max(ensemble_values))
    ks = max(
        abs(long_values.get(threshold, 0.0) - ensemble_values.get(threshold, 0.0))
        for threshold in range(maximum + 1)
    )
    bit_differences = []
    long_samples = sample_count(long_rows)
    for threshold in range(maximum + 1):
        lp = long_values.get(threshold, 0.0)
        ep = ensemble_values.get(threshold, 0.0)
        lc = round(lp * long_samples)
        ec = round(ep * ensemble_samples)
        if lp > 0 and ep > 0 and lc >= 64 and ec >= 64:
            bit_differences.append(abs(-math.log2(lp) + math.log2(ep)))
    epochs = epoch_means(
        os.path.join(BASE, f"fixedgap{gap}_x16_rho{load}_ops24_epochs.csv")
    )
    replicate_means = [run_mean(path) for path in rep_paths]
    return {
        "gap": gap,
        "load_percent": load,
        "long_samples": long_samples,
        "ensemble_samples": ensemble_samples,
        "independent_runs": len(rep_paths),
        "long_mean": mean_from_survival(long_values),
        "ensemble_mean": mean_from_survival(ensemble_values),
        "mean_difference": mean_from_survival(ensemble_values) - mean_from_survival(long_values),
        "ks_distance": ks,
        "max_tail_bit_difference_count_ge_64": max(bit_differences) if bit_differences else math.nan,
        "long_epoch_mean_min": min(epochs),
        "long_epoch_mean_max": max(epochs),
        "long_epoch_mean_sd": statistics.pstdev(epochs),
        "independent_mean_min": min(replicate_means),
        "independent_mean_max": max(replicate_means),
        "independent_mean_sd": statistics.pstdev(replicate_means),
        "long_points": tail_points(long_rows),
        "ensemble_points": points_from_counts(ensemble_samples, ensemble_counts),
    }


def render_comparison(comparisons):
    width, height = 1420, 970
    margin_x, margin_y = 90, 105
    gap_x, gap_y = 70, 90
    panel_w = (width - 2 * margin_x - gap_x) / 2
    panel_h = (height - 2 * margin_y - gap_y) / 2
    x_max = 18.0
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            42,
            "Single long run versus eight independent runs",
            text_anchor="middle",
            font_size="25",
            font_weight="bold",
        ),
        svg_text(
            width / 2,
            71,
            "Pre-batch empirical tails; endpoint loads",
            text_anchor="middle",
            font_size="15",
            fill="#444",
        ),
    ]

    for index, record in enumerate(comparisons):
        column, row = index % 2, index // 2
        left = margin_x + column * (panel_w + gap_x)
        top = margin_y + row * (panel_h + gap_y)
        observed = max(
            [point[1] for point in record["long_points"]]
            + [point[1] for point in record["ensemble_points"]]
        )
        step = 5 if observed <= 50 else 10
        y_max = int(math.ceil((observed + 3) / step) * step)

        def sx(value):
            return left + value / x_max * panel_w

        def sy(value):
            return top + panel_h - value / y_max * panel_h

        for value in range(0, int(x_max) + 1, 3):
            x = sx(value)
            items.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + panel_h}" stroke="#ededed"/>'
            )
            items.append(svg_text(x, top + panel_h + 22, value, text_anchor="middle", font_size="11"))
        for value in range(0, y_max + 1, step):
            y = sy(value)
            items.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_w}" y2="{y:.2f}" stroke="#ededed"/>'
            )
            items.append(svg_text(left - 9, y + 4, value, text_anchor="end", font_size="11"))
        items.append(
            f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#222"/>'
        )
        items.append(
            svg_text(
                left + panel_w / 2,
                top - 15,
                f"gap={record['gap']}, rho={record['load_percent']}%",
                text_anchor="middle",
                font_size="17",
                font_weight="bold",
            )
        )

        long_plot = [
            (sx(x), sy(y)) for x, y, _ in record["long_points"] if x <= x_max
        ]
        ensemble_plot = [
            (sx(x), sy(y)) for x, y, _ in record["ensemble_points"] if x <= x_max
        ]
        items.append(polyline(long_plot, "#111111", width=2.4))
        items.append(polyline(ensemble_plot, "#d55e00", dash="8 5", width=2.4))
        items.append(
            svg_text(
                left + 12,
                top + 20,
                f"mean {record['long_mean']:.2f} vs {record['ensemble_mean']:.2f}; KS={record['ks_distance']:.3f}",
                font_size="12",
                fill="#333",
            )
        )

    legend_y = height - 28
    items.append(
        f'<line x1="{width/2-185}" y1="{legend_y}" x2="{width/2-145}" y2="{legend_y}" stroke="#111" stroke-width="3"/>'
    )
    items.append(svg_text(width / 2 - 135, legend_y + 5, "one 2²⁴-operation run", font_size="13"))
    items.append(
        f'<line x1="{width/2+45}" y1="{legend_y}" x2="{width/2+85}" y2="{legend_y}" stroke="#d55e00" stroke-width="3" stroke-dasharray="8 5"/>'
    )
    items.append(svg_text(width / 2 + 95, legend_y + 5, "8 independent 2²⁰ runs", font_size="13"))
    items.append("</svg>")
    path = os.path.join(OUT, "fixed_gap_time_vs_ensemble.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(items))
    return path


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.9f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def main():
    records = {}
    slope_rows = []
    summary_rows = []
    for gap in GAPS:
        records[gap] = {}
        for load in LOADS:
            path = os.path.join(BASE, f"fixedgap{gap}_x16_rho{load}_ops24.csv")
            rows = read_metric(path)
            records[gap][load] = rows
            slopes = local_slopes(rows)
            summary_rows.append(
                {
                    "gap": gap,
                    "load_percent": load,
                    "operations": 1 << 24,
                    "pre_batch_samples": sample_count(rows),
                    "pre_batch_mean": mean_from_survival(survival(rows)),
                    "pre_batch_max": maximum_from_rows(rows),
                    **slopes,
                }
            )
            for name, value in slopes.items():
                slope_rows.append(
                    {
                        "gap": gap,
                        "load_percent": load,
                        "window": name.removeprefix("slope_bits_"),
                        "blocks_per_probability_bit": value,
                    }
                )

    figures = [render_gap(gap, records[gap]) for gap in GAPS]
    by_load_figure = render_by_load(records)
    comparisons = [
        compare_long_and_ensemble(gap, load)
        for gap in GAPS
        for load in (100, 125)
    ]
    comparison_figure = render_comparison(comparisons)
    comparison_rows = [
        {key: value for key, value in row.items() if not key.endswith("_points")}
        for row in comparisons
    ]
    write_csv(os.path.join(OUT, "fixed_gap_prebatch_summary.csv"), summary_rows)
    write_csv(os.path.join(OUT, "fixed_gap_prebatch_local_slopes.csv"), slope_rows)
    write_csv(os.path.join(OUT, "fixed_gap_time_vs_ensemble.csv"), comparison_rows)
    for path in figures + [by_load_figure, comparison_figure]:
        print(path)


if __name__ == "__main__":
    main()
