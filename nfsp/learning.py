"""Seed-aware learning curves and descriptive, predeclared plateau diagnostics."""
import csv
import json
from pathlib import Path
import re

import numpy as np

from .reports import escape, table


def seed_summary(values, bounds=None):
    values = np.asarray(values, float)
    n, mean = len(values), float(values.mean())
    if n < 2:
        return dict(mean=mean, seeds=n, std=None, ci95=None)
    critical = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447,
                7:2.365, 8:2.306, 9:2.262, 10:2.228, 20:2.086, 30:2.042,
                60:2.000, 120:1.980}
    df = max(k for k in critical if k <= n-1)
    std = float(values.std(ddof=1))
    radius = critical[df] * std / np.sqrt(n)
    ci = [mean-radius, mean+radius]
    if bounds:
        ci = [max(bounds[0], ci[0]), min(bounds[1], ci[1])]
    return dict(mean=mean, seeds=n, std=std, ci95=ci)


def add_learning_arguments(parser):
    parser.add_argument('--plateau-window', type=int, default=5,
                        help='number of final evaluation points in the descriptive plateau window')
    parser.add_argument('--plateau-delta', type=float, default=.02,
                        help='maximum practical win-rate change/range (0.02 = 2 percentage points)')
    parser.add_argument('--plateau-min-rounds', type=int, default=10,
                        help='minimum completed evaluation intervals before plateau analysis')


def learning_protocol(args):
    if args.plateau_window < 4 or args.plateau_min_rounds < args.plateau_window - 1:
        raise ValueError('plateau-window >= 4 and plateau-min-rounds >= window - 1 required')
    if not np.isfinite(args.plateau_delta) or not 0 < args.plateau_delta < 1:
        raise ValueError('plateau-delta must be finite and in (0, 1)')
    return dict(window=args.plateau_window, delta=args.plateau_delta,
                min_rounds=args.plateau_min_rounds,
                rule='Final common checkpoint window; each seed range <= delta and the 95% seed t interval of late-half minus early-half means lies inside [-delta,+delta]. At least 3 seeds. Descriptive only; does not stop training or establish convergence.',
                alignment='Linear interpolation within common observed support per algorithm for time/decisions; no extrapolation. Games use common measured checkpoints. No smoothing.',
                uncertainty='Pointwise 95% Student-t interval across training seeds; not simultaneous bands or deck-level intervals.',
                time='Pure training seconds, excluding evaluation, report writing and checkpoint I/O.')


def _groups(curves, baseline):
    groups = {}
    for row in curves:
        if row['baseline'] == baseline:
            seed = row.get('training_seed', row.get('seed'))
            if seed is None:
                raise ValueError('learning curves require a training seed for every observation')
            groups.setdefault(row['algorithm'], {}).setdefault(seed, []).append(row)
    return groups


def curve_statistics(curves, baseline, axis='games'):
    """Align each seed before aggregation; never turn checkpoints into replications."""
    output = []
    for algorithm, seeds in sorted(_groups(curves, baseline).items()):
        # Older experiment results lack decisions: omit that axis, never fabricate counts.
        if any(axis not in r for rows in seeds.values() for r in rows):
            continue
        series = []
        for rows in seeds.values():
            rows = sorted(rows, key=lambda r: r[axis])
            x = np.asarray([r[axis] for r in rows], float)
            y = np.asarray([r['win_rate'] for r in rows], float)
            if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
                raise ValueError('curve coordinates must be finite and strictly increasing per seed')
            series.append((x, y))
        if axis == 'games':
            grid = sorted(set.intersection(*(set(x) for x, _ in series)))
        else:
            low, high = max(x[0] for x, _ in series), min(x[-1] for x, _ in series)
            grid = sorted({float(v) for x, _ in series for v in x if low <= v <= high})
        for value in grid:
            summary = seed_summary([np.interp(value, x, y) for x, y in series], (0, 1))
            output.append(dict(algorithm=algorithm, baseline=baseline, axis=axis, x=float(value), **summary))
    return output


def learning_chart(curves, baseline, axis='games'):
    stats = curve_statistics(curves, baseline, axis)
    if not stats:
        return f'<p>{escape(baseline)} / {escape(axis)}: 无完整原始计数，未绘图。</p>'
    xmax = max(1e-9, max(r['x'] for r in stats))
    xy = lambda x, y: f'{60+x/xmax*970:.2f},{300-y*250:.2f}'
    palette = ['#55cad2', '#ffa366', '#ae9cff', '#e97baa', '#bcd75e']
    svg = '<svg viewBox="0 0 1100 380" role="img" aria-label="learning curves with pointwise seed confidence intervals">'
    for value in (0, .25, .5, .75, 1):
        y = 300-value*250
        svg += f'<path d="M60 {y}H1060" stroke="#34435d"/><text x="5" y="{y}" fill="#ddd">{value:.0%}</text>'
    for i, algorithm in enumerate(sorted({r['algorithm'] for r in stats})):
        rows = [r for r in stats if r['algorithm'] == algorithm]
        color = palette[i % len(palette)]
        if rows[0]['ci95'] is not None:
            band = [xy(r['x'], r['ci95'][0]) for r in rows]
            band += [xy(r['x'], r['ci95'][1]) for r in reversed(rows)]
            svg += f'<polygon class="seed-ci" points="{" ".join(band)}" fill="{color}" fill-opacity="0.18"/>'
        # Thin lines and measured dots preserve the actual unsmoothed seed trajectories.
        for seed_rows in _groups(curves, baseline)[algorithm].values():
            seed_rows = sorted(seed_rows, key=lambda r: r[axis])
            seed_rows = [r for r in seed_rows if rows[0]['x'] <= r[axis] <= rows[-1]['x']]
            points = ' '.join(xy(r[axis], r['win_rate']) for r in seed_rows)
            svg += f'<polyline class="raw-seed" points="{points}" fill="none" stroke="{color}" stroke-opacity="0.3" stroke-width="1"/>'
            for r in seed_rows:
                svg += f'<circle cx="{60+r[axis]/xmax*970:.2f}" cy="{300-r["win_rate"]*250:.2f}" r="2" fill="{color}" fill-opacity="0.45"/>'
        points = ' '.join(xy(r['x'], r['mean']) for r in rows)
        svg += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        for r in rows:
            ci = 'N/A' if r['ci95'] is None else f'{r["ci95"][0]:.2%} to {r["ci95"][1]:.2%}'
            svg += f'<circle cx="{60+r["x"]/xmax*970:.2f}" cy="{300-r["mean"]*250:.2f}" r="3" fill="{color}"><title>{escape(algorithm)}: {r["x"]:.1f}, {r["mean"]:.2%}; n={r["seeds"]}; pointwise 95% CI {ci}</title></circle>'
        svg += f'<text x="{70+i*200}" y="345" fill="{color}">{escape(algorithm)} (n={rows[0]["seeds"]})</text>'
    for fraction in (0, .25, .5, .75, 1):
        x, value = 60 + fraction * 970, fraction * xmax
        label = f'{value/1_000_000:.2g}M' if value >= 1_000_000 else f'{value/1000:.3g}k' if value >= 1000 else f'{value:.1f}'
        svg += f'<path d="M{x} 300v5" stroke="#8295b0"/><text x="{x}" y="322" fill="#ddd" text-anchor="middle">{label}</text>'
    svg += f'<text x="1030" y="368" fill="#ddd" text-anchor="end">{escape(axis)}</text></svg>'
    return f'<div class="card"><h2>学习曲线 · {escape(baseline)} · {escape(axis)}</h2>{svg}<p>粗线：种子均值；细线/小点：原始种子；阴影：逐点 95% 种子 t 区间（单种子无区间）。时间/决策数在各算法种子共同覆盖范围内线性插值，不外推。时间仅计训练。</p></div>'


def plateau_analysis(curves, protocol):
    results = []
    window, delta = protocol['window'], protocol['delta']
    for baseline in sorted({r['baseline'] for r in curves}):
        for algorithm, seeds in sorted(_groups(curves, baseline).items()):
            checkpoints = {seed: {r['games']: r for r in rows} for seed, rows in seeds.items()}
            common = sorted(set.intersection(*(set(rows) for rows in checkpoints.values())))
            result = dict(algorithm=algorithm, baseline=baseline, seeds=len(seeds),
                          completed_rounds=max(0, len(common)-1), status='insufficient_data',
                          window_start_games=None, window_end_games=None, change=None, per_seed=[])
            if len(common)-1 >= protocol['min_rounds'] and len(common) >= window:
                selected = common[-window:]
                for seed, rows in sorted(checkpoints.items()):
                    values = np.asarray([rows[g]['win_rate'] for g in selected])
                    half = window // 2
                    result['per_seed'].append(dict(seed=seed, change=float(values[-half:].mean()-values[:half].mean()),
                                                   span=float(np.ptp(values))))
                change = seed_summary([r['change'] for r in result['per_seed']])
                result.update(window_start_games=selected[0], window_end_games=selected[-1], change=change)
                if len(seeds) >= 3:
                    bounded = all(r['span'] <= delta + 1e-12 for r in result['per_seed'])
                    equivalent = change['ci95'][0] >= -delta and change['ci95'][1] <= delta
                    result['status'] = 'plateau_observed' if bounded and equivalent else 'plateau_not_established'
            results.append(result)
    return results


def plateau_table(rows, protocol=None):
    labels = dict(insufficient_data='数据不足', plateau_observed='当前窗口观察到平台期',
                  plateau_not_established='未确立平台期')
    body = table(['算法', '对手', '种子数', '共同评估轮数', '窗口（训练局数）', '后半窗−前半窗', '变化的95%种子区间', '判断'],
                 [[r['algorithm'], r['baseline'], r['seeds'], r['completed_rounds'],
                   f'{r["window_start_games"]}–{r["window_end_games"]}' if r['change'] else '—',
                   f'{r["change"]["mean"]:+.2%}' if r['change'] else '—',
                   ('[' + ', '.join(f'{v:+.2%}' for v in r['change']['ci95']) + ']') if r['change'] and r['change']['ci95'] is not None else '—', labels[r['status']]] for r in rows])
    rule = ''
    if protocol:
        rule = f'<p>预设规则：至少 {protocol["min_rounds"]} 轮、3 个种子；末尾 {protocol["window"]} 个共同评估点内，每个种子的胜率极差 ≤ {protocol["delta"]*100:g} 个百分点，且后半窗与前半窗均值差的 95% 种子区间完全位于 ±{protocol["delta"]*100:g} 个百分点内。奇数窗口的中间点只参与极差判断。</p>'
    return '<h2>预先定义的平台期诊断</h2>' + rule + body + '<p>这是给定对手、窗口和评估精度下的描述性判断，不是收敛证明；不显著不等于稳定。本诊断不触发早停，也不用于选择最终测试模型。</p>'


def export_learning(output, curves, protocol):
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    stats = []
    for i, baseline in enumerate(sorted({r['baseline'] for r in curves})):
        for axis in ('games', 'decisions', 'seconds'):
            stats.extend(curve_statistics(curves, baseline, axis))
            card = learning_chart(curves, baseline, axis)
            for svg in re.findall(r'<svg\b.*?</svg>', card, re.S):
                svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" style="background:#19263b;font-family:Arial,sans-serif" ', 1)
                (figures / f'learning_{i}_{axis}.svg').write_text(svg, encoding='utf-8')
    analysis = plateau_analysis(curves, protocol)
    (output / 'learning_statistics.json').write_text(json.dumps(stats, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'plateau.json').write_text(json.dumps(dict(protocol=protocol, results=analysis), indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'learning_statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['algorithm', 'baseline', 'axis', 'x', 'mean', 'seeds', 'std', 'ci95_low', 'ci95_high']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            row = dict(row)
            ci = row.pop('ci95') or [None, None]
            writer.writerow(dict(row, ci95_low=ci[0], ci95_high=ci[1]))
    return analysis


def write_live(output, curves, protocol, complete=False):
    """Atomic offline snapshot: refresh reads only the last complete HTML file."""
    output = Path(output)
    body = '<h1>训练进度 · ' + ('已完成' if complete else '运行中（每次评估后更新）') + '</h1>'
    body += '<p>训练中仅显示已有种子；种子数会变化，最终结论以 report.html 为准。中断时页面可能保留最后状态，请同时检查训练进程。</p>'
    body += f'<p>已记录 {len(curves)} 条评估结果。</p>'
    for baseline in sorted({r['baseline'] for r in curves}):
        for axis in ('games', 'decisions', 'seconds'):
            body += learning_chart(curves, baseline, axis)
    body += plateau_table(plateau_analysis(curves, protocol), protocol)
    refresh = '' if complete else '<meta http-equiv="refresh" content="10">'
    html = '<!doctype html><html lang="zh"><meta charset="utf-8">' + refresh
    html += '<title>Schnapsen 训练进度</title><style>body{background:#101827;color:#dce8f9;font:16px Arial;max-width:1200px;margin:24px auto}svg{width:100%}td,th{padding:8px;border:1px solid #34435d}table{border-collapse:collapse}</style>' + body + '</html>'
    temporary = output / 'live.html.tmp'
    temporary.write_text(html, encoding='utf-8')
    temporary.replace(output / 'live.html')
