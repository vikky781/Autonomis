"""Derived, read-only HTML evidence. Rendering never executes a training path."""

import html
import json
from pathlib import Path


def render_report(ledger, destination, *, summary=None):
    rows = ledger.attempts()
    simulation = ledger.identity["kind"] == "simulation"
    label = ("SIMULATION — fixture outputs, not model performance" if simulation
             else "Recorded campaign evidence — no inferred or invented scores")
    table = []
    search_curve = []
    best = 0.0
    phase_counts = {}
    for row in rows:
        phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
        result = row["result"] or {}
        metrics = result.get("metrics", {})
        auc = metrics.get("roc_auc") if row["phase"] in {"search", "probe"} else None
        ap = metrics.get("average_precision") if auc is not None else None
        reason = row["details"].get("reason", "")
        parent = row["details"].get("parent_id", "")
        alternatives = row["details"].get("alternatives", [])
        if alternatives:
            reason = f"{reason} ({len(alternatives)} alternatives recorded)"
        cells = [row["id"], row["phase"], row["status"], parent,
                 row["details"].get("recipe", {}).get("family", "unavailable"),
                 row["details"].get("recipe", {}).get("backend", "unavailable"),
                 f"{auc:.6f}" if auc is not None else "unavailable",
                 f"{ap:.6f}" if ap is not None else "unavailable",
                 f"{metrics.get('train_roc_auc', 'unavailable')}",
                 result.get("tree_count", "unavailable"),
                 result.get("wall_seconds", "unavailable"), reason, row["error"] or ""]
        table.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
        if row["phase"] == "search" and row["status"] == "succeeded" and auc is not None:
            best = max(best, auc)
            search_curve.append((row["id"], best))
    chart = ""
    if search_curve:
        max_attempt = max(x for x, _ in search_curve)
        points = " ".join(f"{40 + x * 620 / max(1, max_attempt):.1f},{220-y*200:.1f}"
                          for x, y in search_curve)
        chart = (f'<svg viewBox="0 0 700 240" role="img" aria-label="Best search AUC trajectory">'
                 f'<line x1="40" y1="220" x2="660" y2="220" stroke="#7d8b91"/>'
                 f'<line x1="40" y1="20" x2="40" y2="220" stroke="#7d8b91"/>'
                 f'<polyline points="{points}" fill="none" stroke="#176345" stroke-width="3"/>'
                 f'<text x="45" y="238" font-size="12">attempt number</text>'
                 f'<text x="45" y="16" font-size="12">best ROC-AUC</text></svg>')
    final = ("<h2>Locked artifact roles</h2><pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>"
             if summary else "<p>No trained submission artifact is available. Hidden-test score unavailable.</p>")
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Autonomis evidence</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#17232b}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border:1px solid #ccd5d9;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}svg{{max-width:700px}}.notice{{background:#fff3d0;padding:16px}}</style>
<h1>Autonomis experiment evidence</h1><p class="notice">{html.escape(label)}</p>
<p>Consumed attempts: <strong>{ledger.used}/15</strong>. Remaining: {ledger.remaining}.
Failed/interrupted reservations remain consumed. This page does not train models.</p>
<p>Phase counts: {html.escape(json.dumps(phase_counts, sort_keys=True))}</p>
<h2>Search selection trajectory</h2><p>Reused Search estimates; not final-model or hidden-test performance.</p>
{chart}<h2>Chronological attempts</h2><table><thead><tr>
<th>ID</th><th>Phase</th><th>Status</th><th>Parent</th><th>Family</th><th>Backend</th>
<th>Validation AUC</th><th>AP</th><th>Train AUC</th><th>Trees</th><th>Attempt seconds</th>
<th>Decision</th><th>Error</th></tr></thead><tbody>{''.join(table)}</tbody></table>
{final}<p>Backend is recorded per recipe. The public IBM reference is not a verified organizer split.</p></html>"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
