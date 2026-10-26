"""Derived, read-only HTML evidence. Rendering never executes a training path."""

import html
import math
from pathlib import Path

ATTEMPT_BUDGET = 15


def _text(value, fallback="unavailable"):
    return html.escape(str(value if value is not None else fallback), quote=True)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric(value):
    number = _number(value)
    return f"{number:.6f}" if number is not None else "unavailable"


def _seconds(value):
    number = _number(value)
    return f"{number:.2f} s" if number is not None else "unavailable"


def _fraction(value):
    number = _number(value)
    if number is None or not 0 <= number <= 1:
        return "unavailable"
    return f"{number:.0%} of training data"


def _svg_polyline(points, *, label, stroke, css_class, viewbox="0 0 720 230"):
    if not points:
        return ""
    mapped = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<svg class="chart {css_class}" viewBox="{viewbox}" role="img" '
        f'aria-label="{_text(label)}" focusable="false">'
        '<line class="axis" x1="54" y1="190" x2="690" y2="190"/>'
        '<line class="axis" x1="54" y1="24" x2="54" y2="190"/>'
        f'<polyline points="{mapped}" fill="none" stroke="{stroke}" '
        'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )


def _budget_chart(rows):
    if not rows:
        return '<p class="empty-note">Budget stays untouched until an experiment is recorded.</p>'
    attempts = len(rows)
    points = []
    for index in range(1, attempts + 1):
        x = 54 + index * 636 / max(1, ATTEMPT_BUDGET)
        y = 190 - index * 166 / ATTEMPT_BUDGET
        points.append((x, y))
    remaining = ATTEMPT_BUDGET - attempts
    chart = _svg_polyline(points, label="Training attempts consumed toward the 15-attempt limit",
                          stroke="var(--signal)", css_class="budget-trajectory")
    return (chart + f'<p class="chart-caption">{attempts} used · {remaining} remain · '
            f'limit {ATTEMPT_BUDGET}</p>')


def _search_chart(search_curve):
    if not search_curve:
        return '<p class="empty-note">No Search results recorded. The score will appear after a successful Search run.</p>'
    max_attempt = max(attempt for attempt, _ in search_curve)
    points = []
    for attempt, score in search_curve:
        x = 54 + attempt * 636 / max(1, ATTEMPT_BUDGET)
        bounded_score = min(1.0, max(0.0, score))
        y = 190 - bounded_score * 166
        points.append((x, y))
    chart = _svg_polyline(points, label="Best Search ROC-AUC by experiment attempt",
                          stroke="var(--accent)", css_class="search-trajectory")
    return chart + (f'<p class="chart-caption">Best Search ROC-AUC across {len(search_curve)} '
                    f'successful Search result(s); latest point at attempt {max_attempt}.</p>')


def _artifact_panel(summary, audit_lock):
    if not summary:
        status = "Audit identities locked; summary not available" if audit_lock else "No artifact locked"
        return (
            '<div class="artifact-empty"><span class="status-mark" aria-hidden="true">—</span>'
            f'<div><strong>{_text(status)}</strong>'
            '<p>No trained submission artifact or audit proxy metadata is recorded yet.</p></div></div>'
        )

    submission_role = summary.get("submission_role", "submission artifact")
    submission_hash = summary.get("submission_hash")
    proxy_hash = summary.get("audit_metric_applies_to")
    submission_lock_hash = (audit_lock or {}).get("submission")
    proxy_lock_hash = (audit_lock or {}).get("proxy")
    identities_match = (submission_hash == submission_lock_hash and
                        proxy_hash == proxy_lock_hash and bool(audit_lock))
    if identities_match:
        identity_note = "Summary model hashes match the immutable ledger audit lock."
        proxy_status = "Identity locked"
    elif audit_lock:
        identity_note = (
            "Artifact identity mismatch: summary hashes are not verified against the ledger lock. "
            f"Ledger submission {_text(submission_lock_hash)} vs summary {_text(submission_hash)}; "
            f"ledger proxy {_text(proxy_lock_hash)} vs summary {_text(proxy_hash)}."
        )
        proxy_status = "Identity mismatch"
    else:
        identity_note = "Summary hashes are not verified against a ledger audit lock."
        proxy_status = "Not locked"
    audit_metrics = summary.get("audit_metrics") or {}
    audit_score = audit_metrics.get("roc_auc")
    audit_ap = audit_metrics.get("average_precision")
    hidden_score = summary.get("hidden_test_score")
    if hidden_score is None:
        hidden = "Hidden-test metrics: unavailable"
    else:
        hidden = f"Reported hidden-test score: {_metric(hidden_score)}"
    return f"""
      <div class="artifact-grid">
        <article class="artifact-card">
          <span class="eyebrow">Submission artifact</span>
          <h3>{_text(submission_role)}</h3>
          <dl>
            <dt>Path</dt><dd>{_text(summary.get('submission_path'))}</dd>
            <dt>Model hash</dt><dd class="hash">{_text(submission_hash)}</dd>
            <dt>Training fraction</dt><dd>{_text(_fraction(summary.get('submission_training_fraction')))}</dd>
          </dl>
        </article>
        <article class="artifact-card">
          <span class="eyebrow">Audit proxy</span>
          <h3>{_text(proxy_status)}</h3>
          <dl>
            <dt>Path</dt><dd>{_text(summary.get('proxy_path'))}</dd>
            <dt>Model hash</dt><dd class="hash">{_text(proxy_hash)}</dd>
            <dt>Training fraction</dt><dd>{_text(_fraction(summary.get('proxy_training_fraction')))}</dd>
          </dl>
        </article>
      </div>
      <p class="artifact-footnote">{identity_note} Audit ROC-AUC {_metric(audit_score)} and
        average precision {_metric(audit_ap)} apply to proxy
        <code>{_text(proxy_hash)}</code>, not automatically to the full-data submission.
        {_text(hidden)}. Official submission schema remains unverified.</p>
    """


def render_report(ledger, destination, *, summary=None):
    """Render campaign evidence as a standalone page without mutating the ledger."""
    rows = ledger.attempts()
    used = ledger.used
    remaining = max(0, ATTEMPT_BUDGET - used)
    simulation = ledger.identity["kind"] == "simulation"
    audit_lock = ledger.audit_lock
    mode = (
        '<span class="mode-pill simulation">SIMULATION | fixture outputs, not model performance</span>'
        if simulation else
        '<span class="mode-pill recorded">Recorded campaign evidence | values shown only when saved</span>'
    )

    search_curve = []
    search_best = None
    timeline = []
    for row in rows:
        phase = str(row["phase"])
        status = str(row["status"])
        result = row["result"] or {}
        metrics = result.get("metrics", {})
        is_search_metric = phase in {"search", "probe"} and status == "succeeded"
        auc = _number(metrics.get("roc_auc")) if is_search_metric else None
        ap = _number(metrics.get("average_precision")) if is_search_metric else None
        if phase == "search" and auc is not None:
            search_best = auc if search_best is None else max(search_best, auc)
            search_curve.append((int(row["id"]), search_best))
        details = row["details"]
        recipe = details.get("recipe", {})
        reason = details.get("reason", "")
        if details.get("alternatives"):
            reason = f"{reason} ({len(details['alternatives'])} alternatives recorded)"
        metric_text = (
            f"Search ROC-AUC {_metric(auc)} · AP {_metric(ap)}" if phase == "search" and auc is not None else
            f"Probe ROC-AUC {_metric(auc)} · AP {_metric(ap)}" if phase == "probe" and auc is not None else
            "No Search metric recorded" if phase in {"search", "probe"} else
            "See audited proxy metrics below" if phase == "audit" else
            "No validation metric by design" if phase == "final" else
            "No validation metric"
        )
        duration = result.get("wall_seconds", result.get("fit_seconds"))
        attempt_number = int(row["id"])
        timeline.append(f"""
          <article class="attempt" aria-label="Attempt {attempt_number}: {_text(phase)} {_text(status)}">
            <div class="attempt-marker"><span>{attempt_number:02d}</span></div>
            <div class="attempt-card">
              <div class="attempt-head">
                <div><span class="eyebrow">Attempt {attempt_number:02d} · {_text(phase)}</span>
                  <h3>{_text(recipe.get('family', 'unavailable'))}
                    <span class="backend">{_text(recipe.get('backend', 'unavailable'))}</span></h3></div>
                <span class="status-pill status-{_text(status)}">{_text(status)}</span>
              </div>
              <dl class="attempt-facts">
                <div><dt>Parent</dt><dd>{_text(details.get('parent_id'), '—')}</dd></div>
                <div><dt>Metrics</dt><dd>{_text(metric_text)}</dd></div>
                <div><dt>Duration</dt><dd>{_text(_seconds(duration))}</dd></div>
                <div><dt>Tree count</dt><dd>{_text(result.get('tree_count'))}</dd></div>
              </dl>
              <p class="rationale"><span>Decision</span>{_text(reason, 'No rationale recorded')}</p>
              {f'<p class="attempt-error">{_text(row["error"])}</p>' if row['error'] else ''}
            </div>
          </article>
        """)

    best_text = _metric(search_best) if search_best is not None else "—"
    audit_status = "locked" if audit_lock else "not yet locked"
    audit_values = (summary or {}).get("audit_metrics", {})
    audit_score = _number(audit_values.get("roc_auc"))
    audit_ap = _number(audit_values.get("average_precision"))
    audit_text = (f"proxy AUC {_metric(audit_score)} | AP {_metric(audit_ap)}"
                  if audit_score is not None else "no audit score")
    identities_match = bool(
        summary and audit_lock and
        summary.get("submission_hash") == audit_lock.get("submission") and
        summary.get("audit_metric_applies_to") == audit_lock.get("proxy")
    )
    artifact_status = (
        "locked" if identities_match else
        "identity mismatch" if summary and audit_lock else
        "summary unverified" if summary else "not available"
    )
    artifacts = _artifact_panel(summary, audit_lock)
    if summary and summary.get("hidden_test_score") is not None:
        hidden_test = f"Reported hidden-test score: {_metric(summary['hidden_test_score'])}"
    else:
        hidden_test = "Hidden-test metrics: unavailable (official test data and result not supplied)"

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Autonomis · Experiment control room</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #f2f5f2;
      --surface: #ffffff;
      --ink: #142b2d;
      --muted: #597073;
      --line: #d6e0dc;
      --signal: #18766b;
      --accent: #bd4d30;
      --soft-signal: #e3f1ed;
      --soft-warm: #f8e9e1;
      --warning: #f1dfac;
      --mono: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      --sans: Inter, "Segoe UI", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 var(--sans); }}
    .shell {{ width: min(1440px, calc(100% - 48px)); margin: 0 auto; padding: 34px 0 56px; }}
    .masthead {{ display: flex; justify-content: space-between; align-items: center; gap: 20px;
      padding-bottom: 20px; border-bottom: 1px solid var(--line); }}
    .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 750; letter-spacing: .01em; }}
    .brand-mark {{ width: 34px; height: 34px; display: grid; place-items: center; background: var(--ink);
      color: #e3f1ed; font: 700 13px var(--mono); border-radius: 9px 9px 9px 2px; }}
    .mast-meta {{ color: var(--muted); font: 11px var(--mono); letter-spacing: .09em; text-transform: uppercase; }}
    .hero {{ padding: 34px 0 24px; display: grid; grid-template-columns: minmax(0, 1fr) auto;
      align-items: end; gap: 24px; }}
    .eyebrow {{ color: var(--signal); font: 700 10px var(--mono); letter-spacing: .13em; text-transform: uppercase; }}
    h1 {{ max-width: 850px; margin: 7px 0 8px; font-size: clamp(30px, 4vw, 50px); line-height: 1.04;
      letter-spacing: -.045em; font-weight: 760; }}
    .hero p {{ margin: 0; max-width: 700px; color: var(--muted); font-size: 16px; }}
    .mode-pill,.status-pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 7px 11px;
      font: 700 10px var(--mono); letter-spacing: .04em; }}
    .mode-pill {{ align-self: center; max-width: 320px; text-align: center; }}
    .mode-pill.recorded {{ color: #165d54; background: var(--soft-signal); }}
    .mode-pill.simulation {{ color: #80361f; background: var(--soft-warm); }}
    .notice {{ margin: 0 0 22px; padding: 13px 16px; border-left: 3px solid var(--warning);
      background: #fff9e8; color: #5b4b21; font-size: 13px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
    .metric-card {{ min-height: 124px; padding: 17px; background: var(--surface); border: 1px solid var(--line);
      border-radius: 4px 14px 14px 14px; box-shadow: 0 3px 8px #193b3310; }}
    .metric-label {{ color: var(--muted); font-size: 12px; }}
    .metric-value {{ display: block; margin: 12px 0 4px; font: 730 clamp(23px, 2.4vw, 34px)/1 var(--mono);
      letter-spacing: -.07em; }}
    .metric-value.compact {{ font-size: clamp(17px, 1.7vw, 23px); letter-spacing: -.04em; }}
    .metric-hint {{ color: var(--muted); font-size: 11px; }}
    .section {{ margin-top: 34px; }}
    .section-head {{ display: flex; align-items: end; justify-content: space-between; gap: 18px;
      margin-bottom: 13px; }}
    .section-head h2 {{ margin: 0; font-size: 20px; letter-spacing: -.025em; }}
    .section-head p {{ margin: 0; color: var(--muted); font-size: 12px; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .chart-card {{ min-width: 0; background: var(--surface); border: 1px solid var(--line); padding: 19px 20px 15px;
      border-radius: 4px 16px 16px 16px; }}
    .chart-card h3 {{ margin: 0; font-size: 14px; }}
    .chart-card .subhead {{ margin: 3px 0 10px; color: var(--muted); font-size: 11px; }}
    .chart {{ width: 100%; height: auto; min-height: 140px; display: block; overflow: visible; }}
    .axis {{ stroke: var(--line); stroke-width: 1.5; }}
    .chart-caption {{ margin: 0; color: var(--muted); font: 11px var(--mono); }}
    .empty-note {{ min-height: 140px; display: grid; place-items: center; padding: 20px; text-align: center;
      color: var(--muted); background: #f7f9f7; border: 1px dashed var(--line); border-radius: 9px; }}
    .timeline {{ display: grid; gap: 10px; }}
    .attempt {{ display: grid; grid-template-columns: 48px minmax(0, 1fr); gap: 13px; }}
    .attempt-marker {{ display: flex; justify-content: center; position: relative; }}
    .attempt-marker::after {{ content: ""; position: absolute; top: 39px; bottom: -13px; width: 1px; background: var(--line); }}
    .attempt:last-child .attempt-marker::after {{ display: none; }}
    .attempt-marker span {{ z-index: 1; width: 34px; height: 34px; display: grid; place-items: center;
      background: var(--ink); color: #fff; border-radius: 50%; font: 700 11px var(--mono); }}
    .attempt-card {{ min-width: 0; background: var(--surface); border: 1px solid var(--line); padding: 16px 18px;
      border-radius: 4px 13px 13px 13px; }}
    .attempt-head {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; }}
    .attempt-head h3 {{ margin: 4px 0 0; font-size: 16px; text-transform: capitalize; }}
    .backend {{ color: var(--muted); font: 11px var(--mono); text-transform: uppercase; margin-left: 5px; }}
    .status-pill {{ padding: 5px 9px; background: #eef1ef; color: #445958; text-transform: uppercase; }}
    .status-succeeded {{ color: #165d54; background: var(--soft-signal); }}
    .status-failed,.status-interrupted {{ color: #8b3926; background: var(--soft-warm); }}
    .attempt-facts {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 13px;
      margin: 15px 0 11px; }}
    .attempt-facts div {{ min-width: 0; }}
    dt {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }}
    dd {{ margin: 3px 0 0; font: 12px var(--mono); overflow-wrap: anywhere; }}
    .rationale {{ margin: 0; padding-top: 10px; border-top: 1px solid #edf1ee; color: #334c4d; font-size: 12px; }}
    .rationale span {{ margin-right: 9px; color: var(--signal); font: 700 10px var(--mono); text-transform: uppercase; }}
    .attempt-error {{ margin: 9px 0 0; color: #8b3926; font-size: 12px; overflow-wrap: anywhere; }}
    .empty-timeline {{ padding: 27px; color: var(--muted); background: var(--surface); border: 1px dashed var(--line);
      border-radius: 4px 15px 15px 15px; }}
    .metric-notes {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .metric-note {{ padding: 15px 16px; background: var(--surface); border-top: 2px solid var(--line); }}
    .metric-note.search-note {{ border-color: var(--signal); }}
    .metric-note.audit-note {{ border-color: var(--accent); }}
    .metric-note.hidden-note {{ border-color: #a8b8b2; }}
    .metric-note h3 {{ margin: 0 0 4px; font-size: 12px; }}
    .metric-note p {{ margin: 0; color: var(--muted); font-size: 11px; }}
    .artifact-panel {{ padding: 19px; background: var(--ink); color: #f3f7f4; border-radius: 4px 18px 18px 18px; }}
    .artifact-panel .eyebrow {{ color: #b7e7d9; }}
    .artifact-panel h2 {{ margin: 4px 0 15px; font-size: 20px; }}
    .artifact-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 11px; }}
    .artifact-card {{ min-width: 0; padding: 16px; color: var(--ink); background: #f7faf8; border-radius: 8px; }}
    .artifact-card h3 {{ margin: 6px 0 12px; font-size: 15px; text-transform: capitalize; }}
    .artifact-card dl {{ display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 8px 10px; margin: 0; }}
    .artifact-card dd {{ margin: 0; }}
    .artifact-card .hash {{ font-size: 10px; }}
    .artifact-footnote {{ margin: 13px 0 0; color: #d1dfda; font-size: 11px; overflow-wrap: anywhere; }}
    .artifact-footnote code {{ color: #b7e7d9; font: 10px var(--mono); }}
    .artifact-empty {{ display: flex; align-items: start; gap: 13px; padding: 17px; color: var(--ink);
      background: #f7faf8; border-radius: 8px; }}
    .artifact-empty p {{ margin: 4px 0 0; color: var(--muted); font-size: 12px; }}
    .status-mark {{ color: var(--signal); font: 20px var(--mono); }}
    footer {{ margin-top: 31px; padding-top: 15px; border-top: 1px solid var(--line); color: var(--muted);
      font-size: 11px; display: flex; justify-content: space-between; gap: 18px; }}
    @media (max-width: 1050px) {{ .metrics {{ grid-template-columns: repeat(3, 1fr); }} }}
    @media (max-width: 720px) {{
      .shell {{ width: min(100% - 28px, 600px); padding-top: 19px; }}
      .hero {{ grid-template-columns: 1fr; padding-top: 25px; }} .mode-pill {{ justify-self: start; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .charts,.artifact-grid,.metric-notes {{ grid-template-columns: 1fr; }}
      .attempt {{ grid-template-columns: 34px minmax(0, 1fr); gap: 8px; }}
      .attempt-marker span {{ width: 29px; height: 29px; }} .attempt-marker::after {{ top: 32px; }}
      .attempt-facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .artifact-card dl {{ grid-template-columns: 95px minmax(0, 1fr); }}
      .section-head {{ align-items: start; flex-direction: column; }} footer {{ flex-direction: column; }}
    }}
    @media (max-width: 390px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} .metric-card {{ padding: 13px; }} }}
    @media (prefers-reduced-motion: reduce) {{ *,*::before,*::after {{ scroll-behavior: auto !important; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="masthead"><div class="brand"><span class="brand-mark" aria-hidden="true">A/15</span>
      <span>Autonomis</span></div><div class="mast-meta">Experiment control room · read-only</div></header>
    <section class="hero" aria-labelledby="page-title">
      <div><span class="eyebrow">Budget-constrained churn research</span>
        <h1 id="page-title">Make every experiment count.</h1>
        <p>One auditable view of the runs, evidence, and artifacts behind the churn prediction campaign.</p></div>
      {mode}
    </section>
    <p class="notice">This page reads saved campaign records only. It does not train or reserve attempts.
      Official organizer data and submission compatibility remain unverified.</p>
    <section class="metrics" aria-label="Campaign status">
      <article class="metric-card"><span class="metric-label">Training attempts used</span>
        <strong class="metric-value">{used} <span>/ 15</span></strong>
        <span class="metric-hint">Failed and interrupted reservations count.</span></article>
      <article class="metric-card"><span class="metric-label">Attempts remaining</span>
        <strong class="metric-value">{remaining}</strong><span class="metric-hint">Hard limit across all phases.</span></article>
      <article class="metric-card"><span class="metric-label">Best Search ROC-AUC</span>
        <strong class="metric-value">{best_text}</strong><span class="metric-hint">Search validation only · not hidden-test performance.</span></article>
      <article class="metric-card"><span class="metric-label">Audit status</span>
        <strong class="metric-value compact">{_text(audit_status)}</strong>
        <span class="metric-hint">{_text(audit_text)} · applies to the proxy.</span></article>
      <article class="metric-card"><span class="metric-label">Artifact status</span>
        <strong class="metric-value compact">{_text(artifact_status)}</strong>
        <span class="metric-hint">Submission identity and proxy shown below.</span></article>
    </section>
    <section class="section" aria-labelledby="trajectory-title">
      <div class="section-head"><div><span class="eyebrow">Campaign trajectory</span>
        <h2 id="trajectory-title">Budget and Search signal</h2></div>
        <p>Search is exploratory; Audit is a separate sealed evaluation.</p></div>
      <div class="charts">
        <article class="chart-card"><h3>Attempt budget</h3>
          <p class="subhead">Each recorded fit consumes one of the 15 available attempts.</p>
          {_budget_chart(rows)}</article>
        <article class="chart-card"><h3>Best Search ROC-AUC</h3>
          <p class="subhead">Running best across successful Search attempts; probes remain distinct.</p>
          {_search_chart(search_curve)}</article>
      </div>
    </section>
    <section class="section" aria-labelledby="evidence-title">
      <div class="section-head"><div><span class="eyebrow">Read the evidence correctly</span>
        <h2 id="evidence-title">Three different score contexts</h2></div></div>
      <div class="metric-notes">
        <article class="metric-note search-note"><h3>Search metrics</h3>
          <p>Recorded on the Search validation partition to select a recipe. These values describe candidate fits.</p></article>
        <article class="metric-note audit-note"><h3>Audit proxy metrics</h3>
          <p>{_text(audit_text)}. The score belongs to the identified Audit proxy, not automatically to the final artifact.</p></article>
        <article class="metric-note hidden-note"><h3>Hidden-test metrics</h3>
          <p>{_text(hidden_test)}. No hidden-test score or Efficiency Factor is inferred here.</p></article>
      </div>
    </section>
    <section class="section" aria-labelledby="timeline-title">
      <div class="section-head"><div><span class="eyebrow">Immutable ledger order</span>
        <h2 id="timeline-title">Experiment timeline</h2></div>
        <p>{len(rows)} recorded attempt(s) · ordered by ledger ID</p></div>
      <div class="timeline">{''.join(timeline) if timeline else
        '<p class="empty-timeline">No experiment attempts have been recorded. The 15-attempt budget remains untouched.</p>'}</div>
    </section>
    <section class="section artifact-panel" aria-labelledby="artifact-title">
      <span class="eyebrow">Frozen terminal roles</span><h2 id="artifact-title">Locked artifacts</h2>
      {artifacts}
    </section>
    <footer><span>Autonomis · derived from local campaign records</span>
      <span>Public IBM reference is not verified organizer data | Efficiency Factor undefined</span></footer>
  </main>
</body>
</html>"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
