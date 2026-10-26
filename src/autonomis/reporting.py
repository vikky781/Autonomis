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
    for row in rows:
        result = row["result"] or {}
        metrics = result.get("metrics", {})
        auc = metrics.get("roc_auc") if row["phase"] in {"search", "probe"} else None
        ap = metrics.get("average_precision") if auc is not None else None
        reason = row["details"].get("reason", "")
        cells = [row["id"], row["phase"], row["status"],
                 row["details"].get("recipe", {}).get("family", "unavailable"),
                 f"{auc:.6f}" if auc is not None else "unavailable",
                 f"{ap:.6f}" if ap is not None else "unavailable",
                 result.get("wall_seconds", "unavailable"), reason, row["error"] or ""]
        table.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
        if row["phase"] == "search" and row["status"] == "succeeded" and auc is not None:
            best = max(best, auc)
            search_curve.append((len(search_curve), best))
    chart = ""
    if search_curve:
        points = " ".join(f"{20 + x * 640 / max(1, len(search_curve)-1):.1f},{220-y*200:.1f}"
                          for x, y in search_curve)
        chart = (f'<svg viewBox="0 0 700 240" role="img" aria-label="Best search AUC trajectory">'
                 f'<polyline points="{points}" fill="none" stroke="#176345" stroke-width="3"/></svg>')
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
<h2>Search selection trajectory</h2><p>Reused Search estimates; not final-model or hidden-test performance.</p>
{chart}<h2>Chronological attempts</h2><table><thead><tr>
<th>ID</th><th>Phase</th><th>Status</th><th>Family</th><th>Validation AUC</th><th>AP</th>
<th>Attempt seconds</th><th>Decision</th><th>Error</th></tr></thead><tbody>{''.join(table)}</tbody></table>
{final}<p>GPU integration is not implemented. The public IBM reference is not a verified organizer split.</p></html>"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
