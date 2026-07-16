#!/usr/bin/env python3
"""Render a Circuit-ORAM-style stash-tail figure for the overnight runs."""

import csv
import html
import math
import os


BASE = "analysis/lane_oram_security/fixed_gap5_overnight_runs"
OUT = "analysis/lane_oram_security"
LOADS = (110, 120)
COLORS = {110: "#0072b2", 120: "#d55e00"}


def svg_text(x, y, value, **attrs):
    defaults = {"x": x, "y": y, "font-family": "sans-serif"}
    defaults.update(attrs)
    encoded = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(item))}"'
        for key, item in defaults.items()
    )
    return f"<text {encoded}>{html.escape(str(value))}</text>"


def polyline(points, color, dash=None, width=3.0, opacity=1.0):
    if len(points) < 2:
        return ""
    attrs = (
        f'fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"'
    )
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline {attrs} points="{coordinates}"/>'


def read_pre_batch(load):
    path = os.path.join(BASE, f"combined_rho{load}_ops32.csv")
    with open(path, newline="", encoding="utf-8") as source:
        return [
            row
            for row in csv.DictReader(source)
            if row["metric"] == "pre_batch"
        ]


def tail_points(rows):
    points = []
    for row in rows:
        count = int(row["exceed_count"])
        samples = int(row["samples"])
        if 0 < count < samples:
            points.append(
                (-math.log2(count / samples), float(row["threshold"]), count)
            )
    return points


def render(records):
    width, height = 1380, 880
    left, right, top, bottom = 115, 55, 125, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_max, y_max = 30.0, 45.0

    def sx(value):
        return left + value / x_max * plot_width

    def sy(value):
        return top + plot_height - value / y_max * plot_height

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            43,
            "Fixed-gap-5 batched ORAM: pre-batch stash tail",
            text_anchor="middle",
            font_size="26",
            font_weight="bold",
        ),
        svg_text(
            width / 2,
            75,
            "X=16 = 8 pooled Y=2 ORAMs; total bottom capacity 2²⁰",
            text_anchor="middle",
            font_size="16",
            fill="#444",
        ),
        svg_text(
            width / 2,
            99,
            "4 independent runs × 2³⁰ operations per load (2²⁴ warm-up per run)",
            text_anchor="middle",
            font_size="15",
            fill="#555",
        ),
    ]

    for value in range(0, 31, 2):
        x = sx(value)
        items.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e8e8e8"/>'
        )
        items.append(
            svg_text(x, top + plot_height + 29, value, text_anchor="middle", font_size="13")
        )
    for value in range(0, 46, 5):
        y = sy(value)
        items.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e8e8e8"/>'
        )
        items.append(
            svg_text(left - 14, y + 5, value, text_anchor="end", font_size="13")
        )

    items.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#222" stroke-width="1.5"/>',
            svg_text(
                left + plot_width / 2,
                height - 31,
                "log₂(1 / Pr[pre-batch stash > R])",
                text_anchor="middle",
                font_size="19",
            ),
            svg_text(
                30,
                top + plot_height / 2,
                "Pre-batch stash threshold R (blocks)",
                text_anchor="middle",
                font_size="19",
                transform=f"rotate(-90 30 {top + plot_height / 2})",
            ),
        ]
    )

    samples = int(records[LOADS[0]][0]["samples"])
    reliable_bits = math.log2(samples / 64)
    boundary_x = sx(reliable_bits)
    items.append(
        f'<line x1="{boundary_x:.2f}" y1="{top}" x2="{boundary_x:.2f}" y2="{top + plot_height}" stroke="#555" stroke-width="1.4" stroke-dasharray="3 5"/>'
    )
    items.append(
        svg_text(
            boundary_x - 7,
            top + 20,
            f"64-exceedance boundary ({reliable_bits:.2f} bits)",
            text_anchor="end",
            font_size="12",
            fill="#444",
        )
    )

    for load in LOADS:
        points = tail_points(records[load])
        reliable = [(sx(x), sy(y)) for x, y, count in points if count >= 64]
        sparse = [(sx(x), sy(y)) for x, y, count in points if count < 64]
        if reliable and sparse:
            sparse.insert(0, reliable[-1])
        items.append(polyline(reliable, COLORS[load]))
        items.append(
            polyline(sparse, COLORS[load], dash="8 6", opacity=0.76)
        )

    legend_x, legend_y = left + 20, top + 35
    for index, load in enumerate(LOADS):
        y = legend_y + 30 * index
        items.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 42}" y2="{y}" stroke="{COLORS[load]}" stroke-width="3.5"/>'
        )
        items.append(
            svg_text(legend_x + 53, y + 5, f"ρ={load}%", font_size="15")
        )

    items.append(
        svg_text(
            left + plot_width - 14,
            top + plot_height - 16,
            "Solid: ≥64 exceedances; dashed: 1–63 exceedances",
            text_anchor="end",
            font_size="13",
            fill="#555",
        )
    )
    items.append("</svg>")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fixed_gap5_overnight_figure3.svg")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(items))
    return path


def main():
    records = {load: read_pre_batch(load) for load in LOADS}
    print(render(records))


if __name__ == "__main__":
    main()
