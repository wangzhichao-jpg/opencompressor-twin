"""Local static report and flat CSV export; no remote assets or telemetry."""

import csv
import html
import json
from pathlib import Path

from .core import TARGETS


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def scatter(rows, target):
    data = [(row[target + "_observed"], row[target]) for row in rows if target in row]
    if not data:
        return "<p>No accepted predictions; inspect refused/failed points.</p>"
    top = max(v for pair in data for v in pair) * 1.05

    def coord(x, y):
        return 45 + 300 * x / top, 330 - 290 * y / top

    dots = "".join(
        f'<circle cx="{coord(x, y)[0]:.2f}" cy="{coord(x, y)[1]:.2f}" r="3" fill="#176f90"/>' for x, y in data
    )
    return f'<svg viewBox="0 0 380 380" role="img" aria-label="{target} observed versus predicted"><path d="M45 20V330H350 M45 330L345 40" fill="none" stroke="#8b98a5"/>{dots}<text x="155" y="365">Observed (SI)</text><text x="12" y="22">Predicted</text><text x="45" y="349">0</text><text x="300" y="349">{top:.4g}</text></svg>'


def comparison(result, target):
    models = [
        ("Semiempirical (4 shared)", result.get("metrics")),
        ("Affine inputs", result.get("baseline", {}).get("metrics")),
        ("AHRI-style (10/response)", result.get("ahri10_baseline", {}).get("metrics")),
    ]
    values = [(label, stats[target]) for label, stats in models if stats and stats.get(target)]
    if not values:
        return "<p>No scoreable accepted test predictions.</p>"
    maximum = max(max(v["mare_percent"], v["p95_are_percent"]) for _, v in values)
    maximum = max(maximum, 1e-6) * 1.2
    content = []
    for i, (label, stats) in enumerate(values):
        y = 30 + i * 72
        content.append(f'<text x="8" y="{y}">{html.escape(label)}</text>')
        for offset, key, color in [(6, "mare_percent", "#176f90"), (24, "p95_are_percent", "#bc7418")]:
            width = 270 * stats[key] / maximum
            content.append(
                f'<rect x="190" y="{y + offset}" width="{width:.3f}" height="12" fill="{color}"/><text x="{195 + width:.3f}" y="{y + offset + 11}">{stats[key]:.3f}%</text>'
            )
    return (
        '<svg viewBox="0 0 560 270" role="img" aria-label="Test error comparison">'
        + "".join(content)
        + '<text x="190" y="260">Blue: MARE · Amber: P95 ARE</text></svg>'
    )


def report(result, output):
    esc = html.escape
    sections = []
    speed = result.get("source_review", {}).get("speed_definition", "see source manifest")
    speed_note = (
        " (equivalent from rated displacement rate, NOT per-point measured speed)"
        if speed == "nominal_mechanical_rpm"
        else ""
    )
    for title, key in [
        ("Metrics (failures retained)", "metrics"),
        ("Training fit ONLY - NOT validation", "training_fit_only_NOT_validation"),
        ("Source admission review", "source_review"),
        ("Affine baseline", "baseline"),
        ("AHRI-style ten-coefficient baseline", "ahri10_baseline"),
        ("Test metrics by training-input pressure slices", "stratified_metrics"),
        ("Prediction interval", "interval"),
        ("Local parameter diagnostics", "diagnostics"),
        ("Frozen group split", "split"),
        ("Assumptions and unfinished evaluation", "limitations"),
        ("Environment", "environment"),
    ]:
        value = result.get(key)
        if key in {"baseline", "ahri10_baseline"} and value:
            value = {k: v for k, v in value.items() if k != "predictions"}
        sections.append(f"<h2>{title}</h2><pre>{esc(json.dumps(value, ensure_ascii=False, indent=2))}</pre>")
    plots = "".join(
        "<div><h3>" + esc(t) + "</h3>" + scatter(result.get("test_predictions", []), t) + "</div>" for t in TARGETS
    )
    comparisons = "".join(
        "<div><h3>" + esc(t) + " · test errors</h3>" + comparison(result, t) + "</div>" for t in TARGETS
    )
    rows = "".join(
        "<tr><td>"
        + esc(str(p.get("point_id")))
        + "</td><td>"
        + esc(str(p.get("status")))
        + "</td><td>"
        + esc(str(p.get("error", "")))
        + "</td></tr>"
        for p in result.get("test_predictions", [])
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>OpenCompressor-Twin report</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#183044;background:#f7f9fb}}h1,h2{{color:#12516a}}.banner{{padding:18px;border-left:6px solid #bc7418;background:#fff1da;font-weight:700}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;border:1px solid #d8e2e7;border-radius:7px}}.plots{{display:grid;grid-template-columns:1fr 1fr}}svg{{width:100%;background:white}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d8e2e7;padding:7px;text-align:left}}@media(max-width:650px){{.plots{{display:block}}}}</style>
<h1>OpenCompressor-Twin · Engineering preview</h1><p class="banner">{esc(result["evidence_label"])}</p>
<p>Dataset: {esc(result["dataset_id"])} · {esc(str(result["compressor"]))}</p>
<p><strong>Power boundary: {esc(result.get("power_boundary", "see source manifest"))}<br>Speed basis: {esc(speed + speed_note)}</strong></p>
<p>SHA-256: {esc(result["dataset_fingerprint"])}</p><p>Source: {esc(result["source"])}<br>License: {esc(result["license"])}</p>
<div class="plots">{plots}</div><h2>Same frozen test points · point-error comparison</h2><div class="plots">{comparisons}</div>{"".join(sections)}<h2>Every final test point</h2><table><tr><th>Point</th><th>Status</th><th>Failure</th></tr>{rows}</table>
<p>Offline preview only. No certification, safe-envelope, measured-time-saving or manufacturer endorsement claim.</p></html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def curve_svg(summary, target):
    key = target + "_mare_mean_percent"
    valid = [s for s in summary if s[key] is not None]
    if not valid:
        return "<p>No scoreable trials; see retained failures/unavailable cases.</p>"
    low, high = min(s["budget_groups"] for s in valid), max(s["budget_groups"] for s in valid)
    top = max(max(s[key] for s in valid) * 1.1, 1e-6)

    def position(budget, value):
        return 55 + 470 * (budget - low) / max(high - low, 1), 330 - 290 * value / top

    parts = [
        '<path d="M55 30V330H550" fill="none" stroke="#8b98a5"/>',
        f'<text x="5" y="25">MARE (%) · 0 to {top:.3g}</text>',
        '<text x="200" y="365">Measured training groups (initial hull included)</text>',
    ]
    colors = {"semiempirical": "#176f90", "affine": "#539755", "ahri10": "#bc7418"}
    for i, (family, color) in enumerate(colors.items()):
        parts.append(f'<text x="{55 + i * 170}" y="400" fill="{color}">{family}</text>')
        for strategy in dict.fromkeys(s["strategy"] for s in summary):
            series = sorted(
                [s for s in valid if s["family"] == family and s["strategy"] == strategy],
                key=lambda s: s["budget_groups"],
            )
            points = " ".join(
                f"{position(s['budget_groups'], s[key])[0]:.2f},{position(s['budget_groups'], s[key])[1]:.2f}"
                for s in series
            )
            dash = (
                'stroke-dasharray="2 3"'
                if strategy == "d_optimal"
                else ('stroke-dasharray="7 4"' if strategy == "maximin" else "")
            )
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" {dash}/>')
            parts.extend(
                f'<circle cx="{position(s["budget_groups"], s[key])[0]:.2f}" cy="{position(s["budget_groups"], s[key])[1]:.2f}" r="3" fill="{color}"/>'
                for s in series
            )
    for budget in sorted({s["budget_groups"] for s in valid}):
        x, _ = position(budget, 0)
        parts.append(f'<text x="{x - 5:.1f}" y="348">{budget}</text>')
    legend = (
        "Solid: random · Dashed: maximin · Dotted: local D-optimal"
        if any(s["strategy"] == "d_optimal" for s in summary)
        else "Solid: random order · Dashed: input-only maximin"
    )
    parts.append('<text x="55" y="425">' + legend + "</text>")
    parts.append('<text x="55" y="447">Mean across orders; repeat SD in CSV, not an independent-data CI.</text>')
    return '<svg viewBox="0 0 620 470" role="img" aria-label="Fixed-budget learning curve">' + "".join(parts) + "</svg>"


def curve_report(result, output):
    esc = html.escape
    plots = "".join(
        "<section><h2>" + esc(t) + "</h2>" + curve_svg(result["summary"], t) + "</section>" for t in TARGETS
    )
    sections = [("Selection audit", "selection_audit")] if "selection_audit" in result else []
    sections += [
        ("Protocol limitations", "limitations"),
        ("Summary (mean and repeat SD)", "summary"),
        ("Retained unavailable / failed trials", "failures"),
    ]
    details = "".join(
        "<h2>" + title + "</h2><pre>" + esc(json.dumps(result[key], ensure_ascii=False, indent=2)) + "</pre>"
        for title, key in sections
    )
    banner = (
        "LOCAL D-OPTIMAL ADAPTIVE REPLAY · secondary retrospective analysis, NOT new independent blind validation; parameter information is not guaranteed error reduction or measured-time saving"
        if result.get("kind") == "adaptive_replay"
        else "SECONDARY RETROSPECTIVE ANALYSIS · not new independent blind validation, not adaptive information-gain selection, no measured-time-saving claim"
    )
    if result.get("source_kind") == "synthetic":
        banner = "SYNTHETIC SOFTWARE SELF-TEST ONLY · NOT real compressor validation. " + banner
    elif result.get("source_kind") == "manufacturer_map":
        banner = "MANUFACTURER MAP REPRODUCTION ONLY · NOT independent experimental validation. " + banner
    elif result.get("source_kind") == "digitized_figure":
        banner = "DIGITIZED FIGURE ANALYSIS · source nature/extraction uncertainty require review. " + banner
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>OpenCompressor-Twin learning curves</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 24px;background:#f7f9fb;color:#183044}}h1,h2{{color:#12516a}}.banner{{padding:18px;border-left:6px solid #bc7418;background:#fff1da}}.plots{{display:grid;grid-template-columns:1fr 1fr}}svg{{width:100%;background:white}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px}}@media(max-width:650px){{.plots{{display:block}}}}</style>
<h1>OpenCompressor-Twin · fixed-budget learning curves</h1><p class="banner">{esc(banner)}</p>
<p>Dataset: {esc(result["dataset_id"])}<br>Power: {esc(result["power_boundary"])}<br><strong>Speed basis: {esc(result.get("speed_definition", "see source manifest"))} (not a per-point shaft-speed validation)</strong><br>Source: {esc(result["source"])}</p>
<p>Dataset SHA-256: {esc(result["dataset_fingerprint"])}<br>Protocol SHA-256: {esc(result["protocol_sha256"])}</p><div class="plots">{plots}</div>{details}</html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output
