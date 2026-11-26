#!/usr/bin/env python3
"""
fifo_depth.py -- generic FIFO depth calculator and picture tool (time-period method)

Each FIFO depth question uses the same inputs:

    --fw / --fr        write clock and read clock, in MHz
    --burst            items in each burst (B)
    --wr / --rd        the clocks that carry a write, or that can carry a read
    --bursts/--period  optional: repeat the bursts, one burst start each PERIOD ns

Pattern grammar (the same for the two sides):
    every            1 item on each clock ("no idle cycles", "1 clock delay between operations")
    idle:N           N idle clocks between items: 1 item each N + 1 clocks
    duty:D           enable duty cycle D (0.25, 25%, 1/4). The enabled clocks are spread.
    bits:11110000    enable sequence that repeats (use this for enables in groups)
    kofn:K/N         K items at any position in each N-clock frame (worst-case position)
    window:P/H       enabled on each clock for H ns in each P ns (worst-case phase)

Time-period method (each run prints these steps):
    T_A = 1 / fA, T_B = 1 / fB          clock periods
    t_W, t_R                            time for 1 write, time for 1 read
                                        (idle:N gives (N + 1) x T, duty:D gives T / D)
    T_burst = B x t_W                   time to write the burst
    N = floor(T_burst / t_R)            reads in the burst time
    D = B - N                           depth (textbook value)
    T_empty = B x t_R                   start each burst T_empty or more after the last start
    window:P/H on the read side:        T_off = P - H and D = T_off / t_W

Depth values:
    textbook    the time-period value above (ideal FIFO, no clock edges)
    exact       peak number of items at the worst clock phase, edge by edge, ideal FIFO
    async-safe  as exact, with S-flop pointer synchronizers and registered flags. It counts
                from the writer side, so that the full flag never stops a write.
    build       the next power of two that is equal to or more than async-safe
                (Gray-code pointers)

Result:
    POSSIBLE              the reader empties the FIFO before the next burst
    POSSIBLE, NO MARGIN   the average write rate is equal to the read rate
    NOT POSSIBLE          each burst adds items, so no depth prevents an overflow

Examples:
    python3 fifo_depth.py --all                       # the ten cases
    python3 fifo_depth.py --case 3 --plot case3.png
    python3 fifo_depth.py --fw 80 --fr 50 --burst 120 --wr idle:1 --rd idle:3
    python3 fifo_depth.py --fw 100 --fr 100 --burst 160 --wr kofn:80/100 --rd kofn:8/10
    python3 fifo_depth.py --fw 25 --fr 100 --burst 100 --rd window:4000/1000 \\
                          --bursts 6 --period 4000
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from fractions import Fraction as Fr

# --------------------------------------------------------------------------- patterns


def _num(text) -> Fr:
    text = str(text).strip()
    if text.endswith('%'):
        return Fr(text[:-1]) / 100
    return Fr(text)


def _rotate_to_one(bits):
    k = list(bits).index(1)
    bits = list(bits)
    return tuple(bits[k:] + bits[:k])


def parse_pattern(text):
    """Return ('bits', tuple) | ('kofn', (k, n)) | ('window', (P_ns, H_ns))."""
    t = str(text).strip().lower().replace(' ', '')
    if t in ('every', 'b2b', 'none', 'noidle', 'idle:0'):
        return ('bits', (1,))
    kind, _, arg = t.partition(':')
    if kind == 'idle':
        n = int(arg)
        if n < 0:
            raise ValueError('idle:N needs N >= 0')
        return ('bits', tuple([1] + [0] * n))
    if kind == 'duty':
        d = _num(arg)
        if d > 1:
            d = d / 100                      # duty:50 means 50 %
        if not 0 < d <= 1:
            raise ValueError('duty must be in (0, 1]')
        d = d.limit_denominator(64)
        p, q = d.numerator, d.denominator
        bits = [((i + 1) * p) // q - (i * p) // q for i in range(q)]
        return ('bits', _rotate_to_one(bits))
    if kind == 'bits':
        bits = [int(c) for c in arg if c in '01']
        if not bits or not any(bits):
            raise ValueError('bits: pattern needs at least one 1')
        return ('bits', _rotate_to_one(bits))
    if kind == 'kofn':
        k, n = (int(v) for v in arg.split('/'))
        if not 0 < k <= n:
            raise ValueError('kofn:K/N needs 0 < K <= N')
        return ('kofn', (k, n))
    if kind == 'window':
        p, h = (_num(v) for v in arg.split('/'))
        if not 0 < h <= p:
            raise ValueError('window:P/H needs 0 < H <= P (ns)')
        return ('window', (p, h))
    raise ValueError(f'unknown pattern {text!r}')


def fraction_used(pat) -> Fr:
    kind, a = pat
    if kind == 'bits':
        return Fr(sum(a), len(a))
    if kind == 'kofn':
        return Fr(a[0], a[1])
    return a[1] / a[0]


def describe(pat) -> str:
    kind, a = pat
    if kind == 'bits':
        if a == (1,):
            return '1 item each clock'
        if a[0] == 1 and not any(a[1:]):
            return f'1 item each {len(a)} clocks'
        return 'enable sequence ' + ''.join(map(str, a))
    if kind == 'kofn':
        return f'{a[0]} items in each {a[1]} clocks, any position'
    return f'enabled for {_fmt(a[1])} ns in each {_fmt(a[0])} ns'


def _fmt(x, nd=2) -> str:
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f'{x:.{nd}f}'.rstrip('0').rstrip('.')


# --------------------------------------------------------------------------- spec


@dataclass
class Spec:
    fw: Fr                      # MHz
    fr: Fr                      # MHz
    burst: int                  # items per burst
    wr: tuple
    rd: tuple
    bursts: int = 1
    period: Fr | None = None    # ns between burst starts
    name: str = ''
    wr_text: str = ''           # pattern text as given (for the printout)
    rd_text: str = ''

    @property
    def tw(self) -> Fr:         # ns
        return Fr(1000) / self.fw

    @property
    def tr(self) -> Fr:
        return Fr(1000) / self.fr

    @property
    def rate_w_burst(self) -> Fr:
        """Items/ns while a burst is being written (worst case)."""
        kind, _ = self.wr
        if kind in ('kofn', 'window'):
            return 1 / self.tw          # back-to-back inside the burst
        return fraction_used(self.wr) / self.tw

    @property
    def rate_w_avg(self) -> Fr:
        """Long-run write rate, items/ns."""
        if self.bursts > 1 and self.period:
            return Fr(self.burst) / self.period
        kind, _ = self.wr
        if kind == 'kofn':
            return fraction_used(self.wr) / self.tw
        return Fr(0)                    # single burst: no long-run stream

    @property
    def rate_r(self) -> Fr:
        """Long-run read capacity, items/ns."""
        return fraction_used(self.rd) / self.tr


# --------------------------------------------------------------------------- time-period method


def item_time(pat, T: Fr, side: str, text: str = ''):
    """Time for 1 item (ns) and its formula, from the clock period T (ns).

    side 'w' = write side (the time inside the burst), 'r' = read side (the average time)."""
    kind, a = pat
    sym = 'T_A' if side == 'w' else 'T_B'
    t = text.strip().lower().replace(' ', '')
    if kind == 'bits':
        f = Fr(sum(a), len(a))
        if a == (1,):
            return T, sym
        if t.startswith('duty:'):
            return T / f, f'{sym} / {_fmt(f, 3)} (duty)'
        if a[0] == 1 and not any(a[1:]):
            return T * len(a), f'({len(a) - 1} + 1) x {sym}'
        return T / f, f'{sym} / {_fmt(f, 3)} (average of the sequence)'
    if kind == 'kofn':
        k, n = a
        if side == 'w':
            return T, f'{sym} (worst case: no gap between writes)'
        return T * Fr(n, k), f'{n} / {k} x {sym} (average)'
    p, h = a                                    # window
    if side == 'w':
        return T, f'{sym} (inside the window)'
    return T * p / h, f'{sym} x {_fmt(p)} / {_fmt(h)} (average)'


def time_steps(s: Spec) -> dict:
    """All the time-period steps for one specification (exact fractions, ns)."""
    T_A, T_B = s.tw, s.tr
    t_W, f_w = item_time(s.wr, T_A, 'w', s.wr_text)
    t_R, f_r = item_time(s.rd, T_B, 'r', s.rd_text)
    assert t_W == 1 / s.rate_w_burst and t_R == 1 / s.rate_r
    B = s.burst
    T_burst = B * t_W
    st = dict(T_A=T_A, T_B=T_B, t_W=t_W, t_R=t_R, f_w=f_w, f_r=f_r, B=B,
              T_burst=T_burst, T_empty=B * t_R, N=None, T_off=None)
    kind, a = s.rd
    if kind == 'window':                        # the reader is off for T_off in each period
        p, h = a
        st['T_off'] = p - h
        st['D'] = max(1, math.ceil(min(st['T_off'], T_burst) / t_W))
        st['mode'] = 'window'
    else:
        st['N'] = math.floor(T_burst / t_R)
        if t_W >= t_R:                          # the reader is as fast as the writer, or faster
            st['D'] = 1
            st['mode'] = 'reader_keeps_up'
        else:
            st['D'] = B - st['N']
            st['mode'] = 'burst'
    return st


def textbook_depth(s: Spec) -> int:
    """Time-period method: items written in T_burst minus items read in T_burst."""
    return time_steps(s)['D']


# --------------------------------------------------------------------------- exact model


def write_times(s: Spec):
    kind, a = s.wr
    tw = s.tw
    out = []
    for j in range(s.bursts):
        t0 = Fr(0) if j == 0 else s.period * j
        n0 = math.ceil(t0 / tw)
        got = 0
        if kind == 'bits':
            n = n0
            while got < s.burst:
                if a[(n - n0) % len(a)]:
                    out.append(n * tw)
                    got += 1
                n += 1
        elif kind == 'kofn':                 # worst case: first frame back-loaded,
            k, nn = a                        # later frames front-loaded
            f = 0
            while got < s.burst:
                base = n0 + f * nn
                slots = range(base + nn - k, base + nn) if f == 0 else range(base, base + k)
                for n in slots:
                    if got < s.burst:
                        out.append(n * tw)
                        got += 1
                f += 1
        else:
            period, high = a
            n = n0
            while got < s.burst:
                t = n * tw
                if (t - t0) % period < high:
                    out.append(t)
                    got += 1
                n += 1
    if any(b <= a_ for a_, b in zip(out, out[1:])):
        raise ValueError('bursts overlap: --period is shorter than one burst')
    return out


def read_edges(s: Spec, phi: Fr, psi, horizon: Fr, w_first: Fr, w_last: Fr):
    tr = s.tr
    m_count = int((horizon - phi) / tr) + 2
    edges = [phi + m * tr for m in range(m_count)]
    kind, a = s.rd
    if kind == 'bits':
        n = len(a)
        en = [a[(m + psi) % n] == 1 for m in range(m_count)]
    elif kind == 'window':
        period, high = a
        en = [((e - psi) % period) < high for e in edges]
    else:                                   # kofn, adversarial placement per frame
        k, nn = a
        en = [False] * m_count
        first_done = False
        for s0 in range((psi % nn) - nn, m_count, nn):
            idx = list(range(s0, s0 + nn))
            lo_t, hi_t = phi + s0 * tr, phi + (s0 + nn - 1) * tr
            if not first_done and hi_t > w_first:
                chosen = idx[:k]            # spend the slots before data arrives
                first_done = True
            elif lo_t <= w_last <= hi_t:
                chosen = idx[nn - k:]       # push the slots past the last write
            else:
                chosen = idx[:k]
            for m in chosen:
                if 0 <= m < m_count:
                    en[m] = True
    return edges, en


def simulate(s: Spec, writes, edges, en, lr_lat: int, lw_lat: int):
    """lr_lat: an item is poppable at the lr_lat-th read edge after its write.
    lw_lat: a pop is visible to the writer at the lw_lat-th write edge after it (0 = at once)."""
    tw = s.tw
    vis = [bisect.bisect_right(edges, w) + lr_lat - 1 for w in writes]
    reads = []
    rd, n = 0, len(writes)
    for m, e in enumerate(edges):
        if rd >= n:
            break
        if en[m] and vis[rd] <= m:
            reads.append(e)
            rd += 1
    seen = reads if lw_lat == 0 else [(math.floor(r / tw) + lw_lat) * tw for r in reads]
    peak, peak_t, j = 0, Fr(0), 0
    for k, w in enumerate(writes):
        while j < len(seen) and seen[j] <= w:
            j += 1
        occ = k + 1 - j
        if occ > peak:
            peak, peak_t = occ, w
    return peak, peak_t, reads


def _phases(s: Spec):
    tw, tr = s.tw, s.tr
    crit, n = [], 0
    while n < 4096:
        v = (n * tw) % tr
        if v in crit:
            break
        crit.append(v)
        n += 1
    pts = sorted(crit)
    cands = set(pts)
    for i, v in enumerate(pts):
        nxt = pts[i + 1] if i + 1 < len(pts) else tr
        cands.add((v + nxt) / 2)
    return sorted(cands)


def _horizon(s: Spec, writes):
    h = writes[-1] + Fr(len(writes)) / s.rate_r + 64 * s.tr
    if s.rd[0] == 'window':
        h += 2 * s.rd[1][0]
    return h


def worst_case(s: Spec, lr_lat: int, lw_lat: int):
    writes = write_times(s)
    horizon = _horizon(s, writes)
    kind, a = s.rd
    if kind == 'bits':
        psis = list(range(len(a)))
    elif kind == 'kofn':
        psis = list(range(a[1]))
    else:
        period = a[0]
        psis = sorted({(j * period / 8 + i * s.tw / 16) % period
                       for j in range(8) for i in range(16)})
    best = None
    for phi in _phases(s):
        for psi in psis:
            edges, en = read_edges(s, phi, psi, horizon, writes[0], writes[-1])
            pk, pt, reads = simulate(s, writes, edges, en, lr_lat, lw_lat)
            if best is None or pk > best['peak']:
                best = dict(peak=pk, t=pt, phi=phi, psi=psi, reads_done=len(reads),
                            writes=len(writes))
    return best


def sync_latency(stages: int):
    """(read-side, write-side) latency in edges for S-flop sync + registered flags."""
    if stages <= 0:
        return 1, 0
    return stages + 2, stages + 2


def next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def analyse(s: Spec, stages: int = 2):
    exact = worst_case(s, 1, 0)
    lr_lat, lw_lat = sync_latency(stages)
    safe = worst_case(s, lr_lat, lw_lat)
    st = time_steps(s)
    lr, lw = s.rate_r, s.rate_w_burst
    T_burst, T_empty = st['T_burst'], st['T_empty']
    avg_w = s.rate_w_avg
    if avg_w > 0:                       # a long-run flow exists (repeated bursts or K-of-N)
        per = (f'Bursts start each {_fmt(s.period)} ns. T_empty = B x t_R = {_fmt(T_empty)} ns. '
               if s.bursts > 1 and s.period else '')
        if avg_w > lr:
            status, short = 'NOT POSSIBLE', 'NOT POSSIBLE'
            result = (per + 'The average write rate is more than the read rate. Each burst adds items. '
                      'No depth prevents an overflow. Use flow control or a faster reader.'
                      + (f' The depth values are only for the {s.bursts} bursts in the model.'
                         if s.bursts > 1 else ''))
        elif avg_w == lr:
            status, short = 'POSSIBLE, NO MARGIN', 'POSSIBLE, NO MARGIN'
            result = (per + 'The average write rate is equal to the read rate. '
                      '1 extra write in each period causes an overflow.')
        else:
            pct = _fmt(avg_w / lr * 100, 1)
            status, short = 'POSSIBLE', f'POSSIBLE ({pct} % of the read rate)'
            result = per + f'The writes use {pct} % of the read capacity. The FIFO is empty before each burst.'
    elif lw < lr:
        status, short = 'POSSIBLE', 'POSSIBLE, also for a continuous flow'
        result = 't_W is more than t_R: the reader is faster. A continuous flow is also possible.'
    elif lw == lr:
        status, short = 'POSSIBLE', 'POSSIBLE for bursts (continuous flow: 1 shared clock only)'
        result = ('t_W is equal to t_R. Bursts are possible. A continuous flow is possible only with '
                  'one shared clock (NO MARGIN). Two oscillators always drift.')
    else:
        status = 'POSSIBLE'
        short = f'POSSIBLE if bursts start {_fmt(T_empty)} ns or more apart'
        result = (f'Start each burst T_empty = B x t_R = {_fmt(T_empty)} ns or more after the last start. '
                  f'The writer is then idle for {_fmt(T_empty - T_burst)} ns or more after each burst. '
                  'A continuous flow is not possible.')
    return dict(
        name=s.name,
        steps={k: (float(v) if isinstance(v, Fr) else v) for k, v in st.items()},
        rate_w_mps=float(lw * 1000), rate_r_mps=float(lr * 1000),
        burst_ns=float(T_burst), empty_ns=float(T_empty),
        textbook=st['D'], exact=exact['peak'], async_safe=safe['peak'],
        build=next_pow2(safe['peak']), stages=stages,
        status=status, result=result, result_short=short,
        exact_detail={k: (float(v) if isinstance(v, Fr) else v) for k, v in exact.items()},
        safe_detail={k: (float(v) if isinstance(v, Fr) else v) for k, v in safe.items()},
    )


# --------------------------------------------------------------------------- presets

CASES = {
    1: dict(name='fA > fB, no idle cycles', fw=80, fr=50, burst=120, wr='every', rd='every', source='45'),
    2: dict(name='fA > fB, 1 clock delay between operations', fw=80, fr=50, burst=120, wr='every', rd='every', source='45'),
    3: dict(name='fA > fB, idle cycles', fw=80, fr=50, burst=120, wr='idle:1', rd='idle:3', source='83'),
    4: dict(name='fA > fB, enable duty cycles', fw=80, fr=50, burst=120, wr='duty:50%', rd='duty:25%', source='83'),
    5: dict(name='fA < fB, no idle cycles', fw=30, fr=50, burst=120, wr='every', rd='every', source='1'),
    6: dict(name='fA < fB, idle cycles', fw=30, fr=50, burst=120, wr='idle:1', rd='idle:3', source='20'),
    7: dict(name='fA = fB, no idle cycles', fw=30, fr=30, burst=120, wr='every', rd='every', source='0 or 1'),
    8: dict(name='fA = fB, idle cycles', fw=50, fr=50, burst=120, wr='idle:1', rd='idle:3', source='60'),
    9: dict(name='80 writes in each 100 clocks, 8 reads in each 10 clocks', fw=100, fr=100, burst=160,
            wr='kofn:80/100', rd='kofn:8/10', source='32'),
    10: dict(name='fA = fB / 4, en_B period 100 x T_A, 25 % duty', fw=25, fr=100, burst=100,
             wr='every', rd='window:4000/1000', bursts=6, period=4000, source='75'),
}


def spec_from_case(n: int) -> Spec:
    c = CASES[n]
    return Spec(fw=Fr(c['fw']), fr=Fr(c['fr']), burst=c['burst'],
                wr=parse_pattern(c['wr']), rd=parse_pattern(c['rd']),
                bursts=c.get('bursts', 1),
                period=Fr(c['period']) if c.get('period') else None,
                name=f"Case {n}: {c['name']}", wr_text=c['wr'], rd_text=c['rd'])


# --------------------------------------------------------------------------- picture


def plot(s: Spec, res: dict, path: str, strip_ns=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    writes = write_times(s)
    best = worst_case(s, 1, 0)          # exact fractions for the worst phase
    phi, psi = best['phi'], best['psi']
    edges, en = read_edges(s, phi, psi, _horizon(s, writes), writes[0], writes[-1])
    _, _, reads = simulate(s, writes, edges, en, 1, 0)

    fig = plt.figure(figsize=(11, 6.2), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.6], hspace=0.5)
    fig.suptitle(s.name or 'FIFO', fontsize=12, fontweight='bold')
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    wc, rc, gc = '#065A82', '#2A9D8F', '#E76F51'

    # timing strip: one box per clock cycle, filled when it carries an item
    span = strip_ns or float(min(writes[-1] + s.tw, 16 * max(s.tw, s.tr)))
    for row, (period, times, enabled, col, lab) in enumerate([
            (s.tr, edges, en, rc, f'RD {_fmt(s.fr)} MHz'),
            (s.tw, None, None, wc, f'WR {_fmt(s.fw)} MHz')]):
        y = row * 1.2
        if times is None:
            wset = set(writes)
            n = 0
            while n * s.tw < span:
                t = n * s.tw
                on = t in wset
                ax0.add_patch(Rectangle((float(t), y), float(s.tw) * 0.92, 0.8,
                                        fc=col if on else 'white', ec=col, lw=0.8))
                n += 1
        else:
            for t, on in zip(times, enabled):
                if t >= span:
                    break
                ax0.add_patch(Rectangle((float(t), y), float(period) * 0.92, 0.8,
                                        fc=col if on else 'white', ec=col, lw=0.8))
        ax0.text(-span * 0.01, y + 0.4, lab, ha='right', va='center', fontsize=10)
    ax0.set_xlim(0, span)
    ax0.set_ylim(-0.3, 2.3)
    ax0.set_yticks([])
    ax0.set_xlabel('time (ns). Filled box: write, or enabled read clock. Empty box: idle clock.')
    for sp in ('left', 'right', 'top'):
        ax0.spines[sp].set_visible(False)

    # cumulative writes vs reads
    wt = [0.0] + [float(t) for t in writes]
    wy = list(range(0, len(writes) + 1))
    rt = [0.0] + [float(t) for t in reads]
    ry = list(range(0, len(reads) + 1))
    ax1.step(wt, wy, where='post', color=wc, lw=2, label='items written')
    ax1.step(rt, ry, where='post', color=rc, lw=2, label='items read')
    tp = float(best['t'])
    k = bisect.bisect_right([float(t) for t in writes], tp)
    r_at = bisect.bisect_right([float(t) for t in reads], tp)
    ax1.annotate('', xy=(tp, k), xytext=(tp, r_at),
                 arrowprops=dict(arrowstyle='<->', color=gc, lw=2))
    ax1.text(tp, (k + r_at) / 2, f'  peak {k - r_at}', color=gc, va='center',
             fontsize=12, fontweight='bold')
    st = res['steps']
    t0 = float(writes[0])
    wts, rts, ymax = [float(t) for t in writes], [float(t) for t in reads], len(writes)
    marks = [(t0 + st['T_burst'], 'T_burst'), (t0 + st['T_empty'], 'T_empty')]
    if abs(marks[0][0] - marks[1][0]) < 1e-9:
        marks = [(marks[0][0], 'T_burst = T_empty')]
    for i, (x, lab) in enumerate(marks):      # put each label where the lines are not
        ax1.axvline(x, color='#56677A', lw=1, ls='--')
        high = max(bisect.bisect_right(wts, x), bisect.bisect_right(rts, x)) > 0.6 * ymax
        ax1.text(x, ymax * (0.02 if high else 0.95), f' {lab} ', color='#56677A',
                 ha='left' if i == 0 else 'right', va='bottom' if high else 'top', fontsize=9)
    ax1.set_xlabel('time (ns)')
    ax1.set_ylabel('items')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='center right')
    ax1.set_title(f"t_W {_fmt(st['t_W'])} ns   t_R {_fmt(st['t_R'])} ns   "
                  f"T_burst {_fmt(st['T_burst'])} ns   |   textbook {res['textbook']}   exact {res['exact']}   "
                  f"async-safe {res['async_safe']}   build {res['build']}", fontsize=10)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- cli


def _steps_lines(res: dict, s: Spec) -> list:
    st = res['steps']
    f = _fmt
    rows = [('T_A', '1 / fA', f'1 / {f(s.fw)} MHz = {f(st["T_A"])} ns'),
            ('t_W', st['f_w'], f'{f(st["t_W"])} ns'),
            ('T_burst', 'B x t_W', f'{st["B"]} x {f(st["t_W"])} = {f(st["T_burst"])} ns'),
            ('T_B', '1 / fB', f'1 / {f(s.fr)} MHz = {f(st["T_B"])} ns'),
            ('t_R', st['f_r'], f'{f(st["t_R"])} ns')]
    if st['mode'] == 'window':
        p, h = s.rd[1]
        rows += [('T_off', 'P - H', f'{f(p)} - {f(h)} = {f(st["T_off"])} ns (the reader is off)'),
                 ('D', 'min(T_off, T_burst) / t_W',
                  f'{f(min(st["T_off"], st["T_burst"]))} / {f(st["t_W"])} = {st["D"]}')]
    else:
        n = st['N']
        rows.append(('N', 'floor(T_burst / t_R)', f'floor({f(st["T_burst"])} / {f(st["t_R"])}) = {n}'))
        if st['mode'] == 'reader_keeps_up':
            rows.append(('D', 'B - N', f'{st["B"]} - {n} = {st["B"] - n}, so D = 1 '
                                       '(ideal: each item is read before the next write)'))
        else:
            rows.append(('D', 'B - N', f'{st["B"]} - {n} = {st["D"]}'))
    rows.append(('T_empty', 'B x t_R', f'{st["B"]} x {f(st["t_R"])} = {f(st["T_empty"])} ns'))
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    return [f'    {a:<{w1}} = {b:<{w2}} = {c}' for a, b, c in rows]


def _report(res: dict, s: Spec, source=None) -> str:
    lines = [res['name'] or 'FIFO',
             f"  Write : {_fmt(s.fw)} MHz, {describe(s.wr)}",
             f"  Read  : {_fmt(s.fr)} MHz, {describe(s.rd)}",
             f"  Burst : B = {s.burst} items" + (f", {s.bursts} bursts, 1 burst start each {_fmt(s.period)} ns"
                                                 if s.bursts > 1 and s.period else ''),
             '  Time-period calculation']
    lines += _steps_lines(res, s)
    lines += ['  Depth values']
    if source is not None:
        lines.append(f"    source answer         : {source}")
    lines += [f"    textbook (time-period): {res['textbook']}",
              f"    exact, worst phase    : {res['exact']}",
              f"    async-safe ({res['stages']}-flop)   : {res['async_safe']}",
              f"    build (power of 2)    : {res['build']}",
              f"  Result: {res['status']}"]
    lines += textwrap.wrap(res['result'], 92, initial_indent='    ', subsequent_indent='    ')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', type=int, choices=sorted(CASES), help='one of the ten cases')
    ap.add_argument('--all', action='store_true', help='table of the ten cases')
    ap.add_argument('--fw', type=str, help='write clock, MHz')
    ap.add_argument('--fr', type=str, help='read clock, MHz')
    ap.add_argument('--burst', type=int, help='items per burst')
    ap.add_argument('--wr', default='every', help='write pattern (see grammar)')
    ap.add_argument('--rd', default='every', help='read pattern (see grammar)')
    ap.add_argument('--bursts', type=int, default=1, help='number of bursts in the model')
    ap.add_argument('--period', type=str, help='ns between burst starts (with --bursts)')
    ap.add_argument('--sync', type=int, default=2, help='synchronizer flops (0 = ideal FIFO)')
    ap.add_argument('--plot', help='write a PNG: clock view, and items written and read')
    ap.add_argument('--json', action='store_true', help='print JSON instead of text')
    a = ap.parse_args(argv)

    if a.all:
        rows = []
        for n in CASES:
            r = analyse(spec_from_case(n), a.sync)
            r['source'] = CASES[n]['source']
            rows.append(r)
        if a.json:
            print(json.dumps(rows, indent=1, default=str))
            return 0
        print(f"{'case':<5}{'source':>7}{'t_W ns':>9}{'t_R ns':>9}{'T_burst ns':>12}{'N':>6}{'D':>5}"
              f"{'exact':>7}{'async':>7}{'build':>7}  result")
        for n, r in zip(CASES, rows):
            st = r['steps']
            nn = '-' if st['N'] is None else str(st['N'])
            print(f"{n:<5}{r['source']:>7}{_fmt(st['t_W']):>9}{_fmt(st['t_R']):>9}{_fmt(st['T_burst']):>12}"
                  f"{nn:>6}{r['textbook']:>5}{r['exact']:>7}{r['async_safe']:>7}{r['build']:>7}  "
                  f"{r['result_short']}")
        print('t_W, t_R: time for 1 write and 1 read. T_burst = B x t_W. N = floor(T_burst / t_R). '
              'D = B - N (case 10: T_off / t_W).')
        return 0

    if a.case:
        s = spec_from_case(a.case)
        source = CASES[a.case]['source']
    else:
        if not (a.fw and a.fr and a.burst):
            ap.error('give --case N, --all, or --fw, --fr and --burst')
        if a.bursts > 1 and not a.period:
            ap.error('--bursts > 1 needs --period')
        s = Spec(fw=_num(a.fw), fr=_num(a.fr), burst=a.burst, wr=parse_pattern(a.wr),
                 rd=parse_pattern(a.rd), bursts=a.bursts,
                 period=_num(a.period) if a.period else None, wr_text=a.wr, rd_text=a.rd)
        source = None
    res = analyse(s, a.sync)
    print(json.dumps(res, indent=1, default=str) if a.json else _report(res, s, source))
    if a.plot:
        plot(s, res, a.plot)
        print(f'  Picture: {a.plot}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
