#!/usr/bin/env python3
"""Summarize minibucket lane experiments and render dependency-free SVGs."""

import csv
import glob
import math
import os
import re

OUT = "analysis/lane_oram_security"
COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]


def rows(path, metric="post"):
    with open(path, newline="", encoding="utf-8") as source:
        return [r for r in csv.DictReader(source) if r["metric"] == metric]


def summary(path):
    data = rows(path)
    mean = sum(float(r["probability"]) for r in data)
    minimum = next(int(r["threshold"]) for r in data if float(r["probability"]) < 1)
    maximum = max(int(r["threshold"]) for r in data)
    return mean, minimum, maximum


def config(path):
    match = re.search(r"b(\d+)_z(\d+)_y(\d+)", path)
    return tuple(map(int, match.groups()))


def fit_tail(path):
    data = rows(path, "insertion")
    points = []
    measured = []
    for row in data:
        p = float(row["probability"])
        if p > 0:
            bits = -math.log2(p)
            measured.append((bits, int(row["threshold"])))
            if 4 <= bits <= 19:
                points.append((bits, int(row["threshold"])))
    if len(points) < 4:
        raise RuntimeError(f"not enough tail points in {path}")
    mx = sum(x for x, _ in points) / len(points)
    my = sum(y for _, y in points) / len(points)
    denom = sum((x - mx) ** 2 for x, _ in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / denom
    intercept = my - slope * mx
    residual = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in points) / (len(points) - 2))
    return measured, intercept, slope, residual


def svg_start(title, y_label, x_label, width=1050, height=680):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="21">{title}</text>',
        f'<text x="{width/2}" y="665" text-anchor="middle" font-family="sans-serif" font-size="16">{x_label}</text>',
        f'<text x="22" y="335" transform="rotate(-90 22 335)" text-anchor="middle" font-family="sans-serif" font-size="16">{y_label}</text>',
    ]


def scaling_plot(records):
    left, top, width, height = 85, 60, 925, 555
    ymax = max(r[5] for r in records) * 1.08
    x = lambda value: left + (value - 10) / 10 * width
    y = lambda value: top + (1 - value / ymax) * height
    out = svg_start("Minibucket Lane ORAM stash scaling", "mean post-eviction stash", "log₂(N)")
    for tick in range(10, 21, 2):
        px = x(tick); out.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top+height}" stroke="#ddd"/>')
        out.append(f'<text x="{px}" y="640" text-anchor="middle" font-family="sans-serif" font-size="13">{tick}</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="black"/>')
    out.append(f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="black"/>')
    configs = sorted(set((r[0], r[1], r[2]) for r in records))
    for index, cfg in enumerate(configs):
        color = COLORS[index % len(COLORS)]
        pts = sorted((r[3], r[5]) for r in records if r[:3] == cfg)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="' + " ".join(f'{x(a):.2f},{y(b):.2f}' for a,b in pts) + '"/>')
        ly = 82 + index * 24
        out.append(f'<line x1="105" y1="{ly}" x2="137" y2="{ly}" stroke="{color}" stroke-width="4"/>')
        out.append(f'<text x="145" y="{ly+5}" font-family="sans-serif" font-size="13">B={cfg[0]}, Z={cfg[1]}, Y={cfg[2]}</text>')
    out.append('</svg>')
    path = os.path.join(OUT, "minibucket_stash_scaling.svg")
    open(path, "w", encoding="utf-8").write("\n".join(out))


def tail_plot(fits):
    canvas_width, canvas_height = 1280, 735
    left, top, width, height = 105, 92, 835, 555
    xmax = 128
    ymax = math.ceil(max(intercept + slope * xmax for _, _, intercept, slope, _ in fits) / 50) * 50
    x = lambda value: left + value / xmax * width
    y = lambda value: top + (1 - value / ymax) * height
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="640" y="31" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">Minibucket Lane ORAM stash-failure tail at N = 2²⁰</text>',
        '<text x="640" y="57" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#444">Failure means insertion demand exceeds the allocated stash capacity S</text>',
    ]
    for tick in range(0, xmax + 1, 16):
        px = x(tick)
        out.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top+height}" stroke="#e4e4e4"/>')
        out.append(f'<text x="{px}" y="{top+height+24}" text-anchor="middle" font-family="sans-serif" font-size="13">{tick}</text>')
    for tick in range(0, ymax + 1, 50):
        py = y(tick)
        out.append(f'<line x1="{left}" y1="{py}" x2="{left+width}" y2="{py}" stroke="#e4e4e4"/>')
        out.append(f'<text x="{left-13}" y="{py+5}" text-anchor="end" font-family="sans-serif" font-size="13">{tick}</text>')
    target_x = x(96)
    out.append(f'<line x1="{target_x}" y1="{top}" x2="{target_x}" y2="{top+height}" stroke="#b2182b" stroke-width="2" stroke-dasharray="8 5"/>')
    out.append(f'<text x="{target_x-7}" y="{top+18}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#9b1625" font-weight="600">target: per-operation failure ≤ 2⁻⁹⁶</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<text x="{left+width/2}" y="{canvas_height-20}" text-anchor="middle" font-family="sans-serif" font-size="16">Security exponent k, where per-operation overflow probability = 2⁻ᵏ</text>')
    out.append(f'<text x="25" y="{top+height/2}" transform="rotate(-90 25 {top+height/2})" text-anchor="middle" font-family="sans-serif" font-size="16">Stash capacity S (64-byte blocks)</text>')
    for index, (cfg, measured, intercept, slope, _) in enumerate(fits):
        color = COLORS[index % len(COLORS)]
        usable = [p for p in measured if 0 <= p[0] <= 24]
        points = " ".join(f'{x(a):.2f},{y(b):.2f}' for a, b in usable)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for bits, capacity in usable:
            out.append(f'<circle cx="{x(bits):.2f}" cy="{y(capacity):.2f}" r="2.3" fill="{color}"/>')
        start = max(a for a, _ in usable)
        out.append(f'<line x1="{x(start):.2f}" y1="{y(intercept+slope*start):.2f}" x2="{x(xmax):.2f}" y2="{y(intercept+slope*xmax):.2f}" stroke="{color}" stroke-width="2.2" stroke-dasharray="8 6"/>')
        out.append(f'<circle cx="{target_x:.2f}" cy="{y(intercept+slope*96):.2f}" r="4" fill="white" stroke="{color}" stroke-width="2.5"/>')
        ly = 125 + index * 68
        estimate = math.ceil(intercept + slope * 96)
        out.append(f'<line x1="975" y1="{ly}" x2="1010" y2="{ly}" stroke="{color}" stroke-width="4"/>')
        out.append(f'<text x="1020" y="{ly+4}" font-family="sans-serif" font-size="14" font-weight="600">B={cfg[0]}, Z={cfg[1]}, Y={cfg[2]}</text>')
        out.append(f'<text x="1020" y="{ly+24}" font-family="sans-serif" font-size="13">estimated S at 2⁻⁹⁶: {estimate}</text>')
        out.append(f'<text x="1020" y="{ly+42}" font-family="sans-serif" font-size="12" fill="#555">tail slope: {slope:.3f} blocks/bit</text>')
    out.append('<rect x="965" y="78" width="290" height="455" fill="none" stroke="#cfcfcf" rx="5"/>')
    out.append('<text x="980" y="558" font-family="sans-serif" font-size="12" fill="#333">● solid + points: measured (through ≈2⁻²²)</text>')
    out.append('<text x="980" y="579" font-family="sans-serif" font-size="12" fill="#333">– – dashed: log-linear extrapolation</text>')
    out.append('<text x="980" y="608" font-family="sans-serif" font-size="12" fill="#8a1c1c">Extrapolation is not a 96-bit proof.</text>')
    out.append('</svg>')
    path=os.path.join(OUT, "minibucket_figure3a.svg")
    open(path,"w",encoding="utf-8").write("\n".join(out))


def main():
    os.makedirs(OUT, exist_ok=True)
    records=[]
    for path in glob.glob("/tmp/scale_b*_z*_y*_l*.csv"):
        b,z,y=config(path); logn=int(re.search(r"_l(\d+)",path).group(1)); mean,minimum,maximum=summary(path)
        records.append((b,z,y,logn,1<<logn,mean,minimum,maximum))
    with open(os.path.join(OUT,"minibucket_scaling_summary.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["b","z","y","log_n","n","mean","minimum","maximum"]); w.writerows(sorted(records))
    scaling_plot(records)
    fits=[]
    for path in sorted(glob.glob("/tmp/tail_b*_z*_y*_n20.csv")):
        cfg=config(path); measured,intercept,slope,residual=fit_tail(path); fits.append((cfg,measured,intercept,slope,residual))
    with open(os.path.join(OUT,"minibucket_tail_fit_summary.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["b","z","y","intercept","stash_per_security_bit","fit_residual","estimated_stash_2^-96"])
        for cfg,_,a,s,r in fits: w.writerow([*cfg,f"{a:.6f}",f"{s:.6f}",f"{r:.6f}",math.ceil(a+s*96)])
    tail_plot(fits)


if __name__ == "__main__":
    main()
