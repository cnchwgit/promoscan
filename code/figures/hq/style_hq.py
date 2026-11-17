#!/usr/bin/env python3
# coding: UTF-8
"""High-standard figure style: draw at final print size, never rescale.

Rules
  * canvas width = 17.1 cm (6.732 in) double-column, or 8.7 cm single column
  * body font 7 pt at 100% (>= 2 mm at print), annotations >= 6 pt
  * bold panel letters at the axes' top-left corner, no in-figure titles
  * Okabe-Ito palette, no alpha tricks on data marks
  * output PNG 600 dpi + LZW TIFF 600 dpi + vector PDF (Type 42) + SVG (editable text)
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CM = 1 / 2.54
W_DOUBLE = 17.1 * CM          # 6.732 in
W_SINGLE = 8.7 * CM

CB, CO, CG, CGRAY, CPUR = '#0072B2', '#D55E00', '#009E73', '#9aa0a6', '#7B3294'

FS_BASE = 7.0     # ticks / general text
FS_LABEL = 7.5    # axis labels
FS_VAL = 6.0      # value labels (hard floor for 2 mm at print)
FS_LETTER = 9.0   # panel letters
FS_BOX = 6.2      # statistics boxes

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': FS_BASE,
    'axes.titlesize': FS_BASE,
    'axes.labelsize': FS_LABEL,
    'xtick.labelsize': FS_BASE,
    'ytick.labelsize': FS_BASE,
    'legend.fontsize': 6.2,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.4,
    'ytick.major.size': 2.4,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'lines.linewidth': 1.1,
    'patch.linewidth': 0.0,
    'grid.color': '#ececec',
    'grid.linewidth': 0.55,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'savefig.dpi': 600,
})


def new_fig(height_in: float, nrows: int = 2, ncols: int = 2, width_in: float = W_DOUBLE,
            left=0.075, right=0.985, bottom=0.075, top=0.975, wspace=0.34, hspace=0.42,
            width_ratios=None, height_ratios=None):
    fig = plt.figure(figsize=(width_in, height_in))
    gs = fig.add_gridspec(nrows, ncols, left=left, right=right, bottom=bottom, top=top,
                          wspace=wspace, hspace=hspace,
                          width_ratios=width_ratios, height_ratios=height_ratios)
    return fig, gs


def letter(ax, s: str, dx: float = -0.115, dy: float = 1.04):
    """Bold panel letter, outside top-left (journal style)."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold',
            ha='left', va='bottom', color='#111111', clip_on=False)


def tidy(ax, axis='y'):
    ax.grid(axis=axis, lw=0.55, color='#ececec')
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- QA utilities
def _texts(fig):
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        try:
            if not t.get_visible() or not t.get_text().strip():
                continue
            if t.get_window_extent().width <= 0:
                continue
            # skip legend-internal text (legend has its own layout engine)
            parent = t.get_figure()
            if parent is None:
                continue
            out.append(t)
        except Exception:
            continue
    return out


def _drawn_texts(fig, renderer):
    """Text artists that are actually painted: skip tick labels outside the view."""
    tick = {}
    for ax in fig.axes:
        bb = ax.get_window_extent(renderer=renderer)
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            b = t.get_window_extent(renderer=renderer)
            cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            inside = (bb.x0 - 6 <= cx <= bb.x1 + 6) and (bb.y0 - 6 <= cy <= bb.y1 + 6)
            tick[id(t)] = (t, inside)
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) in tick:
            if not tick[id(t)][1]:
                continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            b = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if b.width <= 0 or b.height <= 0:
            continue
        out.append(t)
    return out


def check(fig, name: str, verbose: bool = True):
    """Canvas size, text outside canvas, text-text overlaps (rotated labels excluded)."""
    dpi = fig.dpi
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    problems = {'outside': [], 'overlap': []}

    boxes = []
    for t in _drawn_texts(fig, r):
        b = t.get_window_extent(renderer=r)
        txt = t.get_text().replace('\n', '/')[:26]
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            problems['outside'].append((txt, round(b.x0), round(b.x1), round(b.y0), round(b.y1)))
        if t.get_fontsize() < 0.01 or abs(t.get_rotation() % 180) > 1:
            continue
        boxes.append((txt, b, t.get_fontsize()))

    for (t1, b1, s1), (t2, b2, s2) in itertools.combinations(boxes, 2):
        iw = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        ih = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if iw > 1.5 and ih > 1.5:
            problems['overlap'].append((t1, t2, round(iw * ih, 1),
                                        tuple(round(v) for v in (b1.x0, b1.y0, b1.x1, b1.y1)),
                                        tuple(round(v) for v in (b2.x0, b2.y0, b2.x1, b2.y1))))

    if verbose:
        print(f'[{name}] canvas {W/dpi:.2f} x {H/dpi:.2f} in '
              f'({W/dpi*2.54:.1f} x {H/dpi*2.54:.1f} cm)')
        print(f'  fonts: min {min(s for _, _, s in boxes):.1f} pt, '
              f'max {max(s for _, _, s in boxes):.1f} pt, n_text {len(boxes)}')
        print(f'  outside: {len(problems["outside"])}  overlap: {len(problems["overlap"])}')
        for row in problems['outside'][:8]:
            print('   OUT', row)
        for row in problems['overlap'][:12]:
            print('   OVL', row)
        if os.environ.get('DUMP'):
            for i, ap in enumerate(fig.axes):
                print('   AX%d' % i, [round(v * 100) for v in ap.get_position().bounds])
            for t in _drawn_texts(fig, r):
                b = t.get_window_extent(renderer=r)
                print('   T %-22s %4.0f %4.0f %4.0f %4.0f rot=%g' % (
                    t.get_text().replace('\n', '/')[:22], b.x0, b.y0, b.x1, b.y1,
                    t.get_rotation()))
    return problems


def save(fig, outdir: Path, stem: str, name: str | None = None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = name or stem
    problems = check(fig, name)
    fig.savefig(outdir / f'{stem}.png', dpi=600, facecolor='white')
    fig.savefig(outdir / f'{stem}.pdf', facecolor='white')
    fig.savefig(outdir / f'{stem}.tif', dpi=600, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(outdir / f'{stem}.svg', facecolor='white')   # text stays editable
    print(f'  saved {outdir/stem}.[png|pdf|tif|svg]')
    return problems
