"""Standalone HTML + SVG reports: no plotting dependency, CDN or network required."""
import csv
import html
import json
from pathlib import Path


def escape(value):
    return html.escape(str(value), quote=True)


def write_report(output, data, title, body):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;background:#101827;color:#e5edf9;font:16px system-ui,sans-serif}
main{max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:36px}
p{line-height:1.7;color:#b9c8df}.card{background:#19263b;padding:22px;border-radius:14px;margin:20px 0;overflow:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:12px;border-bottom:1px solid #34435d}
svg{width:100%;height:auto}a{color:#75d8ef}small{color:#b9c8df}button{background:#2b496e;color:white;border:0;border-radius:6px;padding:8px;cursor:pointer}
</style><main>"""
    page += f"<title>{escape(title)}</title><h1>{escape(title)}</h1>" + body
    page += '<p><a href="results.json">原始 JSON</a> · <a href="results.csv">CSV 数据</a></p></main></html>'
    (output / "report.html").write_text(page, encoding="utf-8")


def write_csv(output, rows):
    rows = list(rows)
    if not rows:
        return
    with (Path(output) / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '<div class="card"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def bars(title, labels, values, unit="", intervals=None):
    high = max([1e-9, *values, *([v[1] for v in intervals] if intervals else [])]) * 1.12
    height = 65 + 42 * len(labels)
    svg = f'<svg viewBox="0 0 1100 {height}" role="img" aria-label="{escape(title)}"><title>{escape(title)}</title>'
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 22 + i * 42
        svg += f'<text x="0" y="{y+19}" fill="#dce8f9" font-size="13">{escape(label)}</text>'
        svg += f'<rect x="270" y="{y}" width="{650*value/high:.2f}" height="26" rx="3" fill="#55cad2"><title>{escape(label)}: {value:.4f} {escape(unit)}</title></rect>'
        if intervals:
            low, upper = intervals[i]
            a, b = 270 + 650*max(0, low)/high, 270 + 650*upper/high
            svg += f'<path d="M{a} {y+13}H{b} M{a} {y+6}V{y+20} M{b} {y+6}V{y+20}" stroke="#fff" stroke-width="2"/>'
        svg += f'<text x="{280+650*max(value, intervals[i][1] if intervals else value)/high:.2f}" y="{y+19}" fill="#fff" font-size="13">{value:.2f} {escape(unit)}</text>'
    return '<div class="card"><h2>' + escape(title) + '</h2>' + svg + '</svg></div>'


def heatmap(names, matches):
    lookup = {(r["policy"], r["opponent"]): r["win_rate"] for r in matches}
    cells = '<div class="card"><h2>交叉对战胜率</h2><table><tr><th>行策略 / 列对手</th>'
    cells += ''.join(f'<th>{escape(n)}</th>' for n in names) + '</tr>'
    for a in names:
        cells += f'<tr><th>{escape(a)}</th>'
        for b in names:
            value = lookup.get((a, b))
            if value is None and (b, a) in lookup:
                value = 1 - lookup[b, a]
            if value is None:
                cells += '<td>—</td>'
            else:
                hue = 10 + value * 160
                cells += f'<td style="background:hsl({hue:.0f} 42% 27%)">{value:.1%}</td>'
        cells += '</tr>'
    return cells + '</table><p>反向单元格是同一对战记录的补数；对角线未评估。</p></div>'


def stacked_times(labels, rows):
    keys = ("environment_seconds", "inference_seconds", "update_seconds", "overhead_seconds")
    names = ("Environment", "Inference", "Update", "Other")
    colors = ("#55cad2", "#ae9cff", "#ffa366", "#788ba8")
    totals = [sum(r[k] for k in keys) for r in rows]
    high = max([1e-9, *totals]) * 1.05
    svg = f'<svg viewBox="0 0 1100 {90+42*len(rows)}" role="img" aria-label="Measured time breakdown">'
    for j, (name, color) in enumerate(zip(names, colors)):
        svg += f'<rect x="{270+j*190}" y="10" width="12" height="12" fill="{color}"/><text x="{290+j*190}" y="22" fill="#ddd">{name}</text>'
    for i, (label, row) in enumerate(zip(labels, rows)):
        y, x = 45+i*42, 270.
        svg += f'<text x="0" y="{y+18}" fill="#ddd" font-size="13">{escape(label)}</text>'
        for key, color, name in zip(keys, colors, names):
            width = 740*row[key]/high
            svg += f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="26" fill="{color}"><title>{name}: {row[key]:.4f} seconds</title></rect>'
            x += width
        svg += f'<text x="{x+6:.2f}" y="{y+18}" fill="#fff" font-size="13">{totals[i]:.3f}s</text>'
    return '<div class="card"><h2>Measured time breakdown</h2>'+svg+'</svg></div>'


def percent_interval(interval):
    return "N/A (one seed)" if interval is None else f"[{interval[0]:.2%}, {interval[1]:.2%}]"
