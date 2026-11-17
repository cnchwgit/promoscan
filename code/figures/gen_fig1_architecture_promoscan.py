#!/usr/bin/env python3
# coding: UTF-8
"""ProMoScan architecture — manuscript Figure 1 (2026-09-24).

ProMoScan = **Pro**tein -> **Mo**tif -> **Scan**: one protein-conditioned model
for TF binding in which the protein can influence the prediction ONLY through
the motif (PWM) it predicts plus alpha/beta — a functional bottleneck.

Panel layout (left = the two towers, right = interface -> scan -> prediction):

  (i)   Protein tower        protein seq -> ESM-2 650M (frozen) -> MLP -> z_p
  (ii)  Shared DNA tower     DNA 200 bp -> CBR-KAN backbone -> h in R^256
  (iii) Identity interface   z_p -> motif head -> theta, alpha, beta
                             (theta: softmax over bases -> log-odds PWM)
  (iv)  PWM scan & prediction  DNA one-hot (*) theta -> [M,L] scores ->
                             logit = beta + sum_m alpha_m s_m + w.h -> sigma

Self-contained (no project imports).  Outputs, 300 dpi:
  <repo>/figures/Figure1_architecture.{png,svg,pdf,eps}
  <repo>/manuscript/figures/Figure1_architecture.{png,svg,pdf,eps}

Geometry is verified programmatically: text overflow, box-box overlap,
arrow-box collisions, arrow-arrow crossings, canvas bounds.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

# ---------------------------------------------------------------- palette
# Okabe-Ito (same scheme as the ProtoTF Figure 1 -> visual continuity)
DNA, DNA_F = '#0072B2', '#E8F2FA'
PROT, PROT_F = '#009E73', '#E7F5EF'
GATE, GATE_F = '#CC79A7', '#F6E9F1'        # identity interface (purple)
SCAN, SCAN_F = '#D55E00', '#FCE9DE'        # scan / fusion (orange)
DARK = '#2C3E50'
TXT = '#1A1A1A'
C1_FC, C1_EC = '#F2F8FD', DNA
C2_FC, C2_EC = '#F1F9F5', PROT
C3_FC, C3_EC = '#F7F3FB', GATE
C4_FC, C4_EC = '#FDF6EC', '#C0771B'

XLIM, YLIM = 160.0, 120.0
FIGW = 7.3
FIGH = FIGW * YLIM / XLIM
PT_PER_UNIT = 72.0 * FIGW / XLIM
PAD_BOX = 0.25
GAP = 3.4

# layout constants (data units)
YP = 98.0                    # identity row (protein lane + interface)
YU, YL = 70.0, 47.0          # DNA upper line (-> scan) / lower line (-> fusion)
XC, Y_SCAN, Y_FUSE, Y_PRED = 135.0, 70.0, 47.0, 25.0
Y_ONEHOT = 68.0
X_CHANNEL = 104.0            # (unused corridor constant, kept for scripts)
Y_CORRIDOR = 46.0
CW = 36.0                    # shared width of the right-column boxes

_mfig = None
_max = None


def _ensure_measurer():
    global _mfig, _max
    if _mfig is None:
        _mfig, _max = plt.subplots(figsize=(FIGW, FIGH), dpi=300)
        _max.set_xlim(0, XLIM)
        _max.set_ylim(0, YLIM)
        _max.set_aspect('equal')
    return _mfig, _max


def _extent(s, fs):
    mf, ma = _ensure_measurer()
    t = ma.text(XLIM / 2, YLIM / 2, s, fontsize=fs, ha='center', va='center')
    mf.canvas.draw()
    r = mf.canvas.get_renderer()
    ex = t.get_window_extent(r)
    inv = ma.transData.inverted()
    p0 = inv.transform((ex.x0, ex.y0))
    p1 = inv.transform((ex.x1, ex.y1))
    t.remove()
    return p1[0] - p0[0], p1[1] - p0[1]


def tw(s, fs):
    return _extent(s, fs)[0]


def th(s, fs):
    return _extent(s, fs)[1]


_BOXES = []      # (name, patch, texts)
_ARROWS = []     # (name, [p0, p1], color)
_LABELS = []     # (name, text_artist, panel_rect)
_PANELS = []     # panel rects (for overlap / containment checks)


def spec(name, lines, fs, fc, ec, lw=1.4, ls='-', tc=TXT, min_w=0.0):
    w = max(max(tw(s, fs) for s in lines) + 2.0, min_w)
    hs = [th(s, fs) for s in lines]
    pitch = max(max(hs) * 1.34, 1.55 * fs / PT_PER_UNIT)
    h = (len(lines) - 1) * pitch + max(hs) + 1.2
    return dict(name=name, lines=lines, fs=fs, fc=fc, ec=ec, lw=lw, ls=ls,
                tc=tc, w=w, h=h, pitch=pitch, maxh=max(hs))


def place(ax, s, cx, cy, bold=False):
    w, h = s['w'], s['h']
    x, y = cx - w / 2, cy - h / 2
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.25,rounding_size=0.5',
                       fc=s['fc'], ec=s['ec'], lw=s['lw'],
                       linestyle=s['ls'], zorder=4)
    ax.add_patch(p)
    ytop = y + h - 0.6 - s['maxh'] / 2
    texts = []
    for i, l in enumerate(s['lines']):
        t = ax.text(cx, ytop - i * s['pitch'], l, ha='center', va='center',
                    fontsize=s['fs'], color=s['tc'], zorder=5,
                    weight='bold' if bold else 'normal')
        texts.append(t)
    _BOXES.append((s['name'], p, texts))
    return (x, y, w, h)


def panel(ax, x0, y0, x1, y1, fc, ec, lw=1.3, ls=(0, (6, 4)), label=None,
          lfs=7.3, lcolor='#333333'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle='round,pad=0.4,rounding_size=1.4',
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
    if label:
        for j, line in enumerate(label.split('\n')):
            t = ax.text(x0 + 1.6, y1 - 1.4 - j * 2.7, line, fontsize=lfs,
                        color=lcolor, ha='left', va='top', weight='bold',
                        zorder=6)
            _LABELS.append((f'label:{line[:18]}', t, (x0, y0, x1, y1)))
    _PANELS.append((x0, y0, x1, y1))


def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.3, ms=7.5,
          name='arrow', record=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=9))
    if record:
        _ARROWS.append((name, [(x1, y1), (x2, y2)], color))


def path(ax, pts, color='#555555', lw=1.3, ms=7.5, name='path'):
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        arrow(ax, x1, y1, x2, y2, color=color, lw=lw, ms=ms,
              name=f'{name}#{i}', record=(i == len(pts) - 2))


def note(ax, x, y, s, fs=6.3, color='#777777', ha='center', va='center',
         weight='normal', zorder=10):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
                   weight=weight, zorder=zorder)


# ---- visible-edge helpers (FancyBboxPatch pad expands outward) ----
def vx0(r):
    return r[0] - PAD_BOX


def vx1(r):
    return r[0] + r[2] + PAD_BOX


def vy0(r):
    return r[1] - PAD_BOX


def vy1(r):
    return r[1] + r[3] + PAD_BOX


def cxr(r):
    return r[0] + r[2] / 2


def cyr(r):
    return r[1] + r[3] / 2


# ======================================================================
def build(ax):
    R, S = {}, {}

    def add(key, lines, fs, fc, ec, **kw):
        S[key] = spec(key, lines, fs, fc, ec, **kw)
        return S[key]

    def lane(keys, y, x0=2.0, right_edge=None):
        gap = GAP
        if right_edge is not None:
            tot = sum(S[k]['w'] for k in keys)
            gap = (right_edge - x0 - tot) / max(len(keys) - 1, 1)
        x = x0
        for k in keys:
            R[k] = place(ax, S[k], x + S[k]['w'] / 2, y)
            x += S[k]['w'] + gap

    B, M = 6.7, 6.6          # body / mathtext font sizes

    # ---- (i) protein tower -------------------------------------------
    add('pseq', ['TF protein', 'sequence (DBD)'], B, 'white', PROT, lw=1.7)
    add('esm', ['ESM-2 650M', '(frozen, 1280-d)'], B, PROT_F, PROT)
    add('tower', ['Protein tower (MLP)', '1280 → 512 → 128'], B, PROT_F, PROT)
    add('zp', [r'$z_p \in \mathbb{R}^{128}$'], 7.0, '#CFEBDD', PROT, lw=2.0)
    lane(['pseq', 'esm', 'tower', 'zp'], YP)

    # ---- (ii) shared DNA tower: two lines (upper = encoding, lower = tower)
    add('dseq', ['DNA 200 bp', 'peak window'], B, 'white', DNA, lw=1.7)
    add('onehot', ['DNA one-hot', '4 × L'], M, DNA_F, DNA)
    add('bb', ['CBR-KAN shared backbone', '3-mer embedding 66 → 100',
               'multi-scale CNN ×3 (parallel)', 'BiLSTM ×2 → LayerNorm'],
        B, DNA_F, DNA)
    add('h', [r'$h \in \mathbb{R}^{256}$'], 7.0, '#CFE2F3', DNA, lw=2.0)
    lane(['dseq', 'onehot'], YU)
    lane(['bb', 'h'], YL)

    # ---- (iii) identity interface (right, top) -----------------------
    add('ident', ['Identity interface',
                  'motif head (MLP)',
                  '128 → 256 → M·4·W',
                  r'$\rightarrow\ \theta$ : softmax → log-odds',
                  r'$\alpha \in \mathbb{R}^M$ · $\beta \in \mathbb{R}$ (linear)'],
        6.4, GATE_F, GATE, lw=1.8, min_w=CW)
    R['ident'] = place(ax, S['ident'], XC, YP)

    # ---- (iv) scan / fusion / prediction (right column) --------------
    add('scan', ['PWM scan', r'DNA one-hot $\ast$ $\theta$',
                 r'$\rightarrow$ [M, L] position scores'], 6.5, SCAN_F, SCAN,
        lw=1.7, min_w=CW)
    add('fuse', [r'logit $= \beta + \sum_m \alpha_m\, s_m$',
                 r'$+\ w \cdot h$'], M, SCAN_F, SCAN, lw=1.9, min_w=CW)
    add('pred', [r'$\sigma(\mathrm{logit}) \rightarrow$ binding',
                 'probability'], 6.5, DARK, DARK, lw=1.6, tc='white',
        min_w=CW)
    R['scan'] = place(ax, S['scan'], XC, Y_SCAN)
    R['fuse'] = place(ax, S['fuse'], XC, Y_FUSE)
    R['pred'] = place(ax, S['pred'], XC, Y_PRED)
    return R


# ======================================================================
def wire(ax, R):
    # protein lane
    for a, b in (('pseq', 'esm'), ('esm', 'tower'), ('tower', 'zp')):
        arrow(ax, vx1(R[a]), cyr(R[a]), vx0(R[b]), cyr(R[b]), PROT,
              name=f'{a}->{b}')
    # z_p -> identity interface
    arrow(ax, vx1(R['zp']), cyr(R['zp']), vx0(R['ident']), cyr(R['ident']),
          PROT, lw=1.5, name='zp->ident')
    # DNA upper line: sequence -> one-hot -> scan (straight, same row)
    arrow(ax, vx1(R['dseq']), cyr(R['dseq']), vx0(R['onehot']),
          cyr(R['onehot']), DNA, name='dseq->onehot')
    arrow(ax, vx1(R['onehot']), cyr(R['onehot']), vx0(R['scan']),
          cyr(R['scan']), DNA, lw=1.4, name='onehot->scan')
    # DNA lower line: backbone -> h -> fusion (straight, same row)
    arrow(ax, vx1(R['bb']), cyr(R['bb']), vx0(R['h']), cyr(R['h']), DNA,
          name='bb->h')
    arrow(ax, vx1(R['h']), cyr(R['h']), vx0(R['fuse']), cyr(R['fuse']),
          DNA, lw=1.4, name='h->fuse')
    # the DNA sequence feeds both lines (split downwards into the tower)
    arrow(ax, cxr(R['dseq']), vy0(R['dseq']), cxr(R['dseq']),
          vy1(R['bb']), DNA, lw=1.4, name='dseq->bb')
    # interface -> scan (theta)
    arrow(ax, XC - 5.0, vy0(R['ident']), XC - 5.0, vy1(R['scan']), GATE,
          lw=1.4, name='ident->scan')
    # interface -> fusion (alpha, beta), routed around the right side
    path(ax, [(vx1(R['ident']) - 2.0, vy0(R['ident'])),
              (157.0, vy0(R['ident'])),
              (157.0, cyr(R['fuse'])),
              (vx1(R['fuse']), cyr(R['fuse']))], GATE, lw=1.4, name='ab')
    # scan -> fusion -> prediction
    arrow(ax, XC, vy0(R['scan']), XC, vy1(R['fuse']), SCAN, lw=1.5,
          name='scan->fuse')
    arrow(ax, XC, vy0(R['fuse']), XC, vy1(R['pred']), SCAN, lw=1.5,
          name='fuse->pred')
    note(ax, XC - 1.6, (vy0(R['ident']) + vy1(R['scan'])) / 2, r'$\theta$',
         fs=6.8, color=GATE)
    note(ax, 149.5, 79.0, r'$\alpha$, $\beta$', fs=6.8, color=GATE)
    note(ax, 88.0, cyr(R['fuse']) + 1.8, r'$w$', fs=6.8, color=DNA)


# ======================================================================
def panels(ax, R):
    panel(ax, 0.8, vy0(R['pseq']) - 1.3, vx1(R['zp']) + 1.4,
          vy1(R['pseq']) + 5.0, C2_FC, C2_EC,
          label='(i)  Protein tower', lcolor=PROT)
    panel(ax, 0.8, vy0(R['bb']) - 1.4,
          max(vx1(R['onehot']), vx1(R['h'])) + 1.4,
          vy1(R['dseq']) + 5.0, C1_FC, C1_EC,
          label='(ii)  Shared DNA tower', lcolor=DNA)
    panel(ax, vx0(R['ident']) - 4.6, vy0(R['ident']) - 1.3,
          vx1(R['ident']) + 1.5, vy1(R['ident']) + 5.0, C3_FC, C3_EC,
          label='(iii)  Identity interface', lcolor=GATE)
    panel(ax, vx0(R['pred']) - 4.6, vy0(R['pred']) - 1.4,
          vx1(R['pred']) + 1.5, vy1(R['scan']) + 5.0, C4_FC, C4_EC,
          label='(iv)  Scan & prediction', lcolor='#B7791F')


# ======================================================================
def qa(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = 0

    def ext(patch):
        ex = patch.get_window_extent(r)
        p0 = inv.transform((ex.x0, ex.y0))
        p1 = inv.transform((ex.x1, ex.y1))
        return (p0[0], p0[1], p1[0], p1[1])

    rects = {}
    for name, patch, texts in _BOXES:
        rects[name] = ext(patch)
        x0, y0, x1, y1 = rects[name]
        for t in texts:
            tex = t.get_window_extent(r)
            t0 = inv.transform((tex.x0, tex.y0))
            t1 = inv.transform((tex.x1, tex.y1))
            if (t0[0] < x0 - 0.3 or t1[0] > x1 + 0.3 or
                    t0[1] < y0 - 0.3 or t1[1] > y1 + 0.3):
                bad += 1
                print(f'  [QA] text overflow "{name}": "{t.get_text()}"')
        if x0 < 0 or x1 > XLIM or y0 < 0 or y1 > YLIM:
            bad += 1
            print(f'  [QA] "{name}" outside canvas: {x0:.1f}..{x1:.1f} / '
                  f'{y0:.1f}..{y1:.1f}')
    names = list(rects)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = rects[names[i]], rects[names[j]]
            if (a[0] < b[2] - 0.5 and a[2] > b[0] + 0.5 and
                    a[1] < b[3] - 0.5 and a[3] > b[1] + 0.5):
                bad += 1
                print(f'  [QA] box overlap: "{names[i]}" <-> "{names[j]}"')

    for i in range(len(_PANELS)):
        for j in range(i + 1, len(_PANELS)):
            a, b = _PANELS[i], _PANELS[j]
            if (a[0] < b[2] - 0.5 and a[2] > b[0] + 0.5 and
                    a[1] < b[3] - 0.5 and a[3] > b[1] + 0.5):
                bad += 1
                print(f'  [QA] panel overlap: {a} <-> {b}')
    for bname, rr in rects.items():
        if not any(rr[0] > p[0] - 0.8 and rr[2] < p[2] + 0.8 and
                   rr[1] > p[1] - 0.8 and rr[3] < p[3] + 0.8
                   for p in _PANELS):
            bad += 1
            print(f'  [QA] box "{bname}" is not inside any panel')

    def near_edge(px, py, rr):
        x0, y0, x1, y1 = rr
        if not (x0 - 0.9 <= px <= x1 + 0.9 and y0 - 0.9 <= py <= y1 + 0.9):
            return False
        return (abs(px - x0) < 0.9 or abs(px - x1) < 0.9 or
                abs(py - y0) < 0.9 or abs(py - y1) < 0.9)

    def seg_rect(p, q, rr, shrink):
        x0, y0 = rr[0] + shrink, rr[1] + shrink
        x1, y1 = rr[2] - shrink, rr[3] - shrink
        if x1 <= x0 or y1 <= y0:
            return False
        dx, dy = q[0] - p[0], q[1] - p[1]
        t0, t1 = 0.0, 1.0
        for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]),
                       (-dy, p[1] - y0), (dy, y1 - p[1])):
            if abs(pp) < 1e-12:
                if qq < 0:
                    return False
                continue
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
        return True

    segs = []
    for name, (p, q), color in _ARROWS:
        for bname, rr in rects.items():
            if near_edge(p[0], p[1], rr) or near_edge(q[0], q[1], rr):
                continue
            if seg_rect(p, q, rr, 0.35):
                bad += 1
                print(f'  [QA] arrow "{name}" crosses box "{bname}"')
        segs.append((name, p, q))

    def cross(p1, p2, p3, p4):
        def o(a, b, c):
            v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            return 1 if v > 1e-9 else (-1 if v < -1e-9 else 0)
        d1, d2 = o(p3, p4, p1), o(p3, p4, p2)
        d3, d4 = o(p1, p2, p3), o(p1, p2, p4)
        return d1 * d2 < 0 and d3 * d4 < 0

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            n1, a1, b1 = segs[i]
            n2, a2, b2 = segs[j]
            if n1.split('#')[0] == n2.split('#')[0]:
                continue
            if cross(a1, b1, a2, b2):
                bad += 1
                print(f'  [QA] arrow crossing: "{n1}" x "{n2}"')

    # panel labels: must sit inside their panel and clear of every box
    for lname, t, pr in _LABELS:
        tex = t.get_window_extent(r)
        t0 = inv.transform((tex.x0, tex.y0))
        t1 = inv.transform((tex.x1, tex.y1))
        if (t0[0] < pr[0] + 0.6 or t1[0] > pr[2] - 0.6 or
                t1[1] < pr[1] + 0.6 or t0[1] > pr[3] - 0.6):
            bad += 1
            print(f'  [QA] {lname} outside its panel')
        for bname, br in rects.items():
            if (t0[0] < br[2] - 0.2 and t1[0] > br[0] + 0.2 and
                    t0[1] < br[3] - 0.2 and t1[1] > br[1] + 0.2):
                bad += 1
                print(f'  [QA] {lname} overlaps box "{bname}"')
    if bad == 0:
        print('  [QA] OK — no overflow / overlap / arrow-box collision / '
              'arrow crossing')
    return bad


# ======================================================================
def main():
    del _BOXES[:], _ARROWS[:], _LABELS[:], _PANELS[:]
    fig, ax = plt.subplots(figsize=(FIGW, FIGH), dpi=300)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect('equal')
    ax.axis('off')

    R = build(ax)
    wire(ax, R)
    panels(ax, R)

    bad = qa(fig, ax)
    if bad:
        print(f'[Fig1-ProMoScan] {bad} QA warning(s)')
    for k in sorted(R):
        r = R[k]
        print(f'  {k:8s} visible x {vx0(r):7.2f}..{vx1(r):7.2f}'
              f'   y {vy0(r):6.2f}..{vy1(r):6.2f}')
    root = Path(__file__).resolve().parents[2]
    for outdir in (root / 'figures', root / 'manuscript' / 'figures'):
        outdir.mkdir(parents=True, exist_ok=True)
        for e in ('png', 'svg', 'pdf', 'eps'):
            kw = {'dpi': 300} if e == 'png' else {}
            fig.savefig(outdir / f'Figure1_architecture.{e}',
                        bbox_inches='tight', pad_inches=0.03,
                        facecolor='white', **kw)
        print('saved:', outdir / 'Figure1_architecture.png')
    return bad


if __name__ == '__main__':
    sys.exit(main())
