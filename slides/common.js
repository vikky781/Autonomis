const pptxgen = require('pptxgenjs');

const C = {
  ink: '000814',
  navy: '001D3D',
  blue: '003566',
  gold: 'FFC300',
  yellow: 'FFD60A',
};
const W = 10;
const H = 5.625;
const FONT = 'Aptos';

function base(pres, title, kicker, page, dark = true) {
  const s = pres.addSlide();
  s.background = { color: dark ? C.ink : C.navy };
  s.addText(kicker.toUpperCase(), {
    x: 0.55, y: 0.27, w: 8.4, h: 0.2, fontFace: FONT, fontSize: 9,
    charSpacing: 1.5, color: C.gold, margin: 0,
  });
  s.addText(title, {
    x: 0.55, y: 0.55, w: 8.85, h: 0.63, fontFace: FONT, fontSize: 27,
    bold: true, color: C.yellow, margin: 0, fit: 'shrink',
  });
  s.addShape(pres.ShapeType.line, {
    x: 0.55, y: 1.27, w: 8.9, h: 0,
    line: { color: C.blue, width: 1.2 },
  });
  s.addText('AUTONOMIS  ·  UTKRISHTA 2026  ·  CHALLENGE 1', {
    x: 0.55, y: 5.25, w: 7.9, h: 0.16, fontFace: FONT, fontSize: 7.5,
    color: C.gold, charSpacing: 0.6, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 9.28, y: 5.1, w: 0.42, h: 0.32,
    rectRadius: 0.06, fill: { color: C.blue }, line: { color: C.blue },
  });
  s.addText(String(page), {
    x: 9.28, y: 5.11, w: 0.42, h: 0.28, fontFace: FONT, fontSize: 10,
    bold: true, color: C.yellow, align: 'center', valign: 'mid', margin: 0,
  });
  return s;
}

function text(s, value, x, y, w, h, options = {}) {
  s.addText(value, {
    x, y, w, h, fontFace: FONT, fontSize: 13, color: C.yellow,
    breakLine: false, margin: 0.03, valign: 'mid', fit: 'shrink',
    ...options,
  });
}

function box(pres, s, x, y, w, h, fill = C.navy, line = C.blue) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { color: line, width: 1 },
  });
}

function dot(pres, s, x, y, diameter = 0.12, color = C.gold) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: diameter, h: diameter,
    fill: { color }, line: { color, transparency: 100 },
  });
}

function bar(pres, s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.rect, {
    x, y, w, h, fill: { color: fill }, line: { color: fill, transparency: 100 },
  });
}

function page1(pres) {
  const s = pres.addSlide();
  s.background = { color: C.ink };
  bar(pres, s, 0, 0, 0.18, H, C.gold);
  text(s, 'UTKRISHTA 2026  /  AI·ML HACKATHON', 0.7, 0.64, 6.4, 0.25,
    { fontSize: 10, charSpacing: 1.8, color: C.gold });
  text(s, 'AUTONOMIS', 0.68, 1.26, 8.5, 0.9,
    { fontSize: 48, bold: true, color: C.yellow });
  text(s, 'Make every experiment count.', 0.72, 2.22, 6.1, 0.55,
    { fontSize: 24, bold: true, color: C.gold });
  text(s, 'A budget-aware, auditable experiment controller for customer churn prediction.',
    0.72, 2.95, 6.05, 0.72, { fontSize: 14, color: C.yellow, valign: 'top' });
  box(pres, s, 0.72, 4.14, 4.65, 0.75, C.navy, C.blue);
  text(s, 'TEAM  AUTONOMIS', 0.94, 4.34, 2.3, 0.22,
    { fontSize: 12, bold: true, color: C.yellow });
  text(s, 'CHALLENGE 1  ·  BUDGET-CONSTRAINED CHURN', 3.55, 4.36, 5.5, 0.2,
    { fontSize: 9, color: C.gold, align: 'right' });
  text(s, '15 successful CPU fits · all 15 shared attempts used',
    0.74, 5.13, 8.6, 0.22, { fontSize: 9, color: C.gold });
  // One marker per completed fit in the shared allocation.
  for (let i = 0; i < 15; i += 1) {
    const x = 7.15 + (i % 5) * 0.42;
    const y = 1.12 + Math.floor(i / 5) * 0.43;
    s.addShape(pres.ShapeType.ellipse, {
      x, y, w: 0.23, h: 0.23,
      fill: { color: C.gold },
      line: { color: C.gold, width: 1.2 },
    });
  }
  text(s, '15 / 15', 7.14, 2.62, 2.2, 0.55,
    { fontSize: 31, bold: true, color: C.yellow, align: 'center' });
  text(s, 'allocation complete', 7.16, 3.22, 2.15, 0.22,
    { fontSize: 10, color: C.gold, align: 'center' });
  return s;
}

function page2(pres) {
  const s = base(pres, 'The challenge is better experiments, not more experiments', '01 · Problem', 2);
  text(s, 'Retention teams need a useful ranking of churn risk to prioritize outreach. The challenge is finding a reliable model when every fit consumes scarce budget.',
    0.7, 1.58, 5.05, 1.0, { fontSize: 15, valign: 'top' });
  const items = [
    ['15 FITS MAX', 'Every fit-and-validation cycle counts. Failed or interrupted reservations still use a slot.'],
    ['IMBALANCED TARGET', 'ROC-AUC is primary; Average Precision adds a precision–recall view.'],
    ['HIDDEN EVALUATION', 'Organizer-held labels and the official output schema must stay separate from local selection.'],
  ];
  items.forEach((it, i) => {
    const y = 1.55 + i * 1.05;
    box(pres, s, 6.05, y, 3.3, 0.83, C.navy, C.blue);
    dot(pres, s, 6.27, y + 0.22, 0.16, i === 0 ? C.gold : C.yellow);
    text(s, it[0], 6.55, y + 0.12, 2.5, 0.2,
      { fontSize: 10, bold: true, color: C.gold });
    text(s, it[1], 6.55, y + 0.36, 2.55, 0.34,
      { fontSize: 9, color: C.yellow, valign: 'top' });
  });
  text(s, 'PROJECT RESPONSE', 0.72, 3.08, 3.5, 0.2,
    { fontSize: 10, charSpacing: 1.2, color: C.gold });
  text(s, 'Treat the experiment log as a first-class product: proposals have a rationale, reservations are durable, and the Audit is protected from tuning.',
    0.72, 3.43, 4.7, 0.82, { fontSize: 15, color: C.yellow, valign: 'top' });
  box(pres, s, 0.72, 4.48, 8.63, 0.43, C.blue, C.blue);
  text(s, 'Judging target: predictive quality × disciplined use of the experiment budget',
    0.92, 4.57, 8.2, 0.18, { fontSize: 11, bold: true, align: 'center' });
  return s;
}

function page3(pres) {
  const s = base(pres, 'Public reference data, explicit provenance', '02 · Data & preprocessing', 3);
  const stats = [['7,043', 'customer rows'], ['1,869', 'positive churn labels'], ['19', 'predictor fields'], ['21', 'source columns']];
  stats.forEach((item, i) => {
    const x = 0.62 + i * 2.32;
    box(pres, s, x, 1.55, 2.05, 0.82, C.navy, C.blue);
    text(s, item[0], x + 0.14, 1.61, 1.78, 0.46,
      { fontSize: 23, bold: true, color: C.yellow, align: 'center' });
    text(s, item[1], x + 0.1, 2.11, 1.85, 0.18,
      { fontSize: 8.5, color: C.gold, align: 'center' });
  });
  text(s, 'LOCAL PROCESSING FLOW', 0.68, 2.78, 4.2, 0.2,
    { fontSize: 10, charSpacing: 1.1, color: C.gold });
  const steps = [
    ['01', 'Fingerprint', 'Verify source bytes'],
    ['02', 'Validate', 'Schema + IDs'],
    ['03', 'Split', 'Train / Search / Audit'],
    ['04', 'Transform', 'Fit preprocessing on Train only'],
  ];
  steps.forEach((it, i) => {
    const x = 0.66 + i * 2.29;
    box(pres, s, x, 3.15, 2.0, 0.99, C.navy, C.blue);
    text(s, it[0], x + 0.14, 3.28, 0.45, 0.24,
      { fontSize: 13, bold: true, color: C.gold });
    text(s, it[1], x + 0.62, 3.28, 1.22, 0.24,
      { fontSize: 12, bold: true, color: C.yellow });
    text(s, it[2], x + 0.15, 3.68, 1.72, 0.3,
      { fontSize: 9, color: C.gold, valign: 'top' });
    if (i < steps.length - 1) text(s, '›', x + 2.02, 3.49, 0.22, 0.26,
      { fontSize: 19, bold: true, color: C.gold, align: 'center' });
  });
  text(s, 'Profile finding: 11 blank TotalCharges values · no duplicate customer IDs',
    0.7, 4.2, 8.45, 0.14, { fontSize: 7.8, color: C.gold, align: 'center' });
  box(pres, s, 0.68, 4.42, 8.66, 0.49, C.blue, C.blue);
  text(s, 'SHA-256  16320c9c1ec7…448e3055e91   ·   public IBM reference only; organizer equivalence is unverified',
    0.83, 4.57, 8.36, 0.16, { fontSize: 8.6, color: C.yellow, align: 'center' });
  return s;
}

function page4(pres) {
  const s = base(pres, 'An end-to-end pipeline with budget and audit gates', '03 · System design', 4);
  const nodes = [
    ['DATA', 'CSV + hash'], ['SPLITS', 'Immutable rows'], ['PLANNER', 'Next recipe + reason'],
    ['GATE', 'Shared ledger'], ['WORKER', 'CPU / optional GPU'], ['EVIDENCE', 'Metrics + bundle'],
  ];
  nodes.forEach((n, i) => {
    const row = i < 3 ? 0 : 1;
    const col = i % 3;
    const x = 0.76 + col * 3.05;
    const y = 1.74 + row * 1.25;
    box(pres, s, x, y, 2.45, 0.82, i === 3 ? C.blue : C.navy, i === 3 ? C.gold : C.blue);
    text(s, n[0], x + 0.15, y + 0.13, 2.15, 0.18,
      { fontSize: 10, bold: true, color: C.gold, align: 'center', charSpacing: 1 });
    text(s, n[1], x + 0.15, y + 0.42, 2.15, 0.2,
      { fontSize: 11, color: C.yellow, align: 'center' });
  });
  // Directional connectors make the flow readable without implying extra components.
  [[3.22,2.13,0.4,0],[6.27,2.13,0.4,0],[7.9,2.6,0,0.36],
   [6.27,3.38,0.4,0],[3.22,3.38,0.4,0]].forEach(([x,y,w,h]) => {
    s.addShape(pres.ShapeType.line, { x, y, w, h, line: { color: C.gold, width: 1.4, beginArrowType: 'none', endArrowType: 'triangle' } });
  });
  box(pres, s, 0.76, 4.42, 8.6, 0.48, C.navy, C.blue);
  text(s, 'CONTROL GUARANTEES', 0.93, 4.55, 1.6, 0.17,
    { fontSize: 9, bold: true, color: C.gold });
  text(s, 'Atomic reservations  ·  one worker at a time  ·  append-only evidence  ·  round-scoped Audit lock',
    2.6, 4.53, 6.45, 0.2, { fontSize: 9.2, color: C.yellow, align: 'right' });
  return s;
}

function page5(pres) {
  const s = base(pres, 'A finite feedback planner keeps each run interpretable', '04 · Core algorithm', 5);
  const stages = [
    ['OBSERVE', 'Read saved metrics, status, and prior proposal.'],
    ['PROPOSE', 'Change a bounded recipe choice; explain why.'],
    ['RESERVE', 'Atomically claim one shared attempt token.'],
    ['LEARN', 'Record outcome; use it to guide the next proposal.'],
  ];
  stages.forEach((it, i) => {
    const y = 1.5 + i * 0.79;
    dot(pres, s, 0.82, y + 0.12, 0.2, i === 2 ? C.yellow : C.gold);
    if (i < stages.length - 1) s.addShape(pres.ShapeType.line, {
      x: 0.92, y: y + 0.32, w: 0, h: 0.47, line: { color: C.blue, width: 1.5 },
    });
    text(s, it[0], 1.22, y + 0.02, 1.35, 0.22,
      { fontSize: 10, bold: true, color: C.gold });
    text(s, it[1], 2.65, y, 3.05, 0.42,
      { fontSize: 11, color: C.yellow, valign: 'top' });
  });
  box(pres, s, 6.05, 1.54, 3.22, 2.62, C.navy, C.blue);
  text(s, 'IMPLEMENTED MODEL OPTIONS', 6.28, 1.78, 2.72, 0.22,
    { fontSize: 9, bold: true, color: C.gold, align: 'center' });
  ['Logistic Regression', 'CatBoost', 'LightGBM'].forEach((label, i) => {
    const y = 2.22 + i * 0.48;
    bar(pres, s, 6.42, y + 0.04, 0.08, 0.21, i === 0 ? C.gold : C.blue);
    text(s, label, 6.62, y, 2.25, 0.26, { fontSize: 11, color: C.yellow });
  });
  text(s, 'SAVED FITS', 6.28, 3.75, 1.0, 0.18,
    { fontSize: 8, color: C.gold });
  text(s, '15 / 15 Logistic Regression · CPU', 6.28, 3.95, 2.7, 0.18,
    { fontSize: 9, color: C.yellow });
  box(pres, s, 0.82, 4.62, 8.45, 0.34, C.blue, C.blue);
  text(s, 'Feature ablation in V1: log-tenure · active add-on count · historical monthly charge',
    1.0, 4.71, 8.1, 0.14, { fontSize: 9, color: C.yellow, align: 'center' });
  return s;
}

function page6(pres) {
  const s = base(pres, 'Shared 15-slot ledger: every attempt completed', '05 · Experiment history', 6);
  const rows = [
    ['01', 'SEARCH · DONE', '0.863893'], ['02', 'SEARCH · DONE', '0.864002'],
    ['03', 'SEARCH · DONE', '0.868706'], ['04', 'SEARCH · DONE', '0.868936'],
    ['05', 'SEARCH · DONE', '0.869660'], ['06', 'PROBE · DONE', '0.845710'],
    ['07', 'PROBE · DONE', '0.845896'], ['08', 'AUDIT · DONE', '0.851950'],
    ['09', 'FINAL · DONE', '—'],
    ['10', 'SEARCH · DONE', '0.851599'], ['11', 'SEARCH · DONE', '0.851860'],
    ['12', 'SEARCH · DONE', '0.858804'], ['13', 'SEARCH · DONE', '0.859300'],
    ['14', 'AUDIT · DONE', '0.853652'], ['15', 'FINAL · DONE', '—'],
  ];
  text(s, 'SLOT', 0.85, 1.48, 0.45, 0.2, { fontSize: 8, bold: true, color: C.gold });
  text(s, 'PHASE / STATUS', 1.45, 1.48, 1.55, 0.2, { fontSize: 8, bold: true, color: C.gold });
  text(s, 'ROC-AUC', 3.75, 1.48, 1.05, 0.2,
    { fontSize: 8, bold: true, color: C.gold, align: 'right' });
  text(s, 'SLOT', 5.43, 1.48, 0.48, 0.2, { fontSize: 8, bold: true, color: C.gold });
  text(s, 'PHASE / STATUS', 6.0, 1.48, 1.6, 0.2, { fontSize: 8, bold: true, color: C.gold });
  text(s, 'ROC-AUC', 8.23, 1.48, 1.05, 0.2,
    { fontSize: 8, bold: true, color: C.gold, align: 'right' });
  rows.forEach((r, i) => {
    const col = Math.floor(i / 8);
    const row = i % 8;
    const x = 0.7 + col * 4.58;
    const y = 1.73 + row * 0.39;
    box(pres, s, x, y, 4.18, 0.31, C.navy, C.blue);
    text(s, r[0], x + 0.15, y + 0.1, 0.45, 0.2,
      { fontSize: 9, bold: true, color: C.yellow });
    text(s, r[1], x + 0.7, y + 0.08, 2.1, 0.16,
      { fontSize: 7.2, color: r[1].includes('AVAILABLE') ? C.gold : C.yellow });
    text(s, r[2], x + 2.95, y + 0.07, 1.05, 0.17,
      { fontSize: 8.4, bold: true, color: C.yellow, align: 'right' });
  });
  box(pres, s, 0.72, 4.97, 8.62, 0.23, C.blue, C.blue);
  text(s, 'All 15 shared slots have saved fits across two rounds. Audit scores are proxy metrics; final fits have no holdout score.',
    0.86, 5.01, 8.34, 0.14, { fontSize: 8, color: C.yellow, align: 'center' });
  return s;
}

function page7(pres) {
  const s = base(pres, 'Selection evidence is separated from Audit evidence', '06 · Evaluation method', 7);
  const split = [
    ['TRAIN', 60, C.blue], ['SEARCH', 20, C.gold], ['AUDIT', 20, C.yellow],
  ];
  let x = 0.82;
  split.forEach(([label, pct, color]) => {
    const w = 7.9 * pct / 100;
    bar(pres, s, x, 1.75, w, 0.55, color);
    text(s, `${label}  ${pct}%`, x + 0.04, 1.92, w - 0.08, 0.18,
      { fontSize: 9, bold: true, color: C.ink, align: 'center' });
    x += w;
  });
  text(s, 'PROTOCOL', 0.84, 2.67, 2.0, 0.2,
    { fontSize: 9, bold: true, color: C.gold, charSpacing: 1 });
  const points = [
    'Train-only preprocessing protects validation partitions from leakage.',
    'ROC-AUC measures ranking across thresholds; AP captures precision as recall rises.',
    'The Audit proxy is frozen before its held-out evaluation and cannot guide further tuning.',
  ];
  points.forEach((p, i) => {
    const y = 3.05 + i * 0.48;
    dot(pres, s, 0.9, y + 0.05, 0.12, i === 2 ? C.yellow : C.gold);
    text(s, p, 1.18, y, 7.8, 0.28, { fontSize: 10.5, color: C.yellow });
  });
  box(pres, s, 0.82, 4.63, 8.1, 0.36, C.navy, C.blue);
  text(s, 'Final candidate P uses all labelled development rows; it has no internal holdout score.',
    1.02, 4.73, 7.7, 0.15, { fontSize: 9.5, color: C.gold, align: 'center' });
  return s;
}

function page8(pres) {
  const s = base(pres, 'Recorded local performance — with the right context', '07 · Results', 8);
  text(s, 'ROC-AUC · 0–1', 0.82, 1.53, 3.5, 0.2,
    { fontSize: 9, bold: true, color: C.gold });
  const bars = [
    ['Search · round 1', 0.869660, C.gold],
    ['Search · round 2', 0.859300, C.yellow],
    ['Audit · round 1', 0.851950, C.gold],
    ['Audit · round 2', 0.853652, C.yellow],
    ['Chance reference', 0.5, C.blue],
  ];
  bars.forEach((b, i) => {
    const y = 1.76 + i * 0.49;
    text(s, b[0], 0.84, y, 1.42, 0.22, { fontSize: 8.3, color: C.yellow });
    bar(pres, s, 2.25, y + 0.02, 2.75, 0.19, C.blue);
    bar(pres, s, 2.25, y + 0.02, 2.75 * b[1], 0.19, b[2]);
    text(s, b[1].toFixed(6), 5.08, y, 0.92, 0.22,
      { fontSize: 8.4, bold: true, color: C.yellow, align: 'right' });
  });
  text(s, 'Reference only: random-ranking ROC-AUC = 0.50; not a trained baseline.',
    0.84, 4.29, 5.18, 0.3, { fontSize: 8, color: C.gold, valign: 'top' });
  box(pres, s, 6.2, 1.65, 3.08, 2.88, C.navy, C.blue);
  text(s, 'AVERAGE PRECISION', 6.43, 1.91, 2.6, 0.2,
    { fontSize: 9, bold: true, color: C.gold, align: 'center' });
  text(s, 'Search · round 1', 6.48, 2.42, 1.2, 0.2, { fontSize: 8.4, color: C.yellow });
  text(s, '0.695799', 7.82, 2.39, 1.08, 0.28,
    { fontSize: 17, bold: true, color: C.yellow, align: 'right' });
  text(s, 'Search · round 2', 6.48, 2.93, 1.2, 0.2, { fontSize: 8.4, color: C.yellow });
  text(s, '0.702953', 7.82, 2.9, 1.08, 0.28,
    { fontSize: 17, bold: true, color: C.yellow, align: 'right' });
  text(s, 'Audit · round 1', 6.48, 3.44, 1.2, 0.2, { fontSize: 8.4, color: C.yellow });
  text(s, '0.668741', 7.82, 3.41, 1.08, 0.28,
    { fontSize: 17, bold: true, color: C.yellow, align: 'right' });
  text(s, 'Audit · round 2 AP', 6.48, 3.82, 1.35, 0.18, { fontSize: 7.8, color: C.yellow });
  text(s, '0.672107', 7.82, 3.78, 1.08, 0.24,
    { fontSize: 14, bold: true, color: C.yellow, align: 'right' });
  text(s, 'Separate splits. Audit metrics apply only to each round’s locked proxy.',
    6.47, 4.14, 2.58, 0.32, { fontSize: 7.3, color: C.gold, align: 'center' });
  box(pres, s, 0.72, 4.83, 8.62, 0.29, C.blue, C.blue);
  text(s, 'No organizer hidden-test result or Efficiency Factor is available.',
    0.9, 4.9, 8.22, 0.15, { fontSize: 9.1, bold: true, color: C.yellow, align: 'center' });
  return s;
}

function page9(pres) {
  const s = base(pres, 'Demo: run control paired with saved evidence', '08 · Product walkthrough', 9);
  // Dashboard panel: schematic reconstructed from the local app's current content.
  box(pres, s, 0.63, 1.52, 5.45, 3.48, C.navy, C.blue);
  text(s, 'EXPERIMENT CONTROL ROOM', 0.86, 1.73, 4.9, 0.2,
    { fontSize: 9, bold: true, color: C.gold, charSpacing: 0.9 });
  text(s, 'Choose how to train', 0.86, 2.07, 4.7, 0.32,
    { fontSize: 17, bold: true, color: C.yellow });
  box(pres, s, 0.85, 2.58, 2.25, 0.88, C.ink, C.blue);
  box(pres, s, 3.36, 2.58, 2.4, 0.88, C.ink, C.gold);
  text(s, 'MANUAL', 1.02, 2.75, 1.9, 0.18,
    { fontSize: 9, bold: true, color: C.gold });
  text(s, 'One Search fit', 1.02, 3.02, 1.9, 0.2, { fontSize: 11, color: C.yellow });
  text(s, 'SEQUENTIAL', 3.55, 2.75, 2.05, 0.18,
    { fontSize: 9, bold: true, color: C.gold });
  text(s, 'Run phase gates in order', 3.55, 3.02, 2.03, 0.2, { fontSize: 9, color: C.yellow });
  text(s, '15 / 15 used', 0.87, 3.74, 1.85, 0.47,
    { fontSize: 22, bold: true, color: C.yellow });
  text(s, '6 fits this round · 0 remain', 3.24, 3.87, 2.26, 0.2,
    { fontSize: 11, color: C.gold });
  bar(pres, s, 0.87, 4.32, 4.75, 0.13, C.blue);
  bar(pres, s, 0.87, 4.32, 4.75, 0.13, C.gold);
  text(s, 'Illustrative dashboard layout · live values are read from campaign records',
    0.87, 4.68, 4.85, 0.17, { fontSize: 7.3, color: C.gold });
  // Actual saved final functional-check probabilities, identifiers masked for the slide.
  box(pres, s, 6.34, 1.52, 2.99, 3.48, C.navy, C.blue);
  text(s, 'SAVED PREDICTION SAMPLE', 6.55, 1.75, 2.58, 0.2,
    { fontSize: 8.5, bold: true, color: C.gold, align: 'center' });
  text(s, 'REFERENCE OUTPUT · PROVISIONAL', 6.55, 2.1, 2.58, 0.18,
    { fontSize: 7.2, color: C.yellow, align: 'center' });
  const preds = [['Acct 01', '0.722'], ['Acct 02', '0.034'], ['Acct 03', '0.340'], ['Acct 04', '0.042'], ['Acct 05', '0.763']];
  preds.forEach((p, i) => {
    const y = 2.52 + i * 0.39;
    if (i % 2 === 0) bar(pres, s, 6.56, y - 0.03, 2.55, 0.32, C.ink);
    text(s, p[0], 6.68, y, 1.2, 0.18, { fontSize: 9, color: C.yellow });
    text(s, p[1], 8.0, y, 0.88, 0.18, { fontSize: 9, color: C.gold, align: 'right' });
  });
  text(s, 'Probability of churn', 6.56, 4.63, 2.55, 0.18,
    { fontSize: 8, color: C.gold, align: 'center' });
  return s;
}

function page10(pres) {
  const s = base(pres, 'A strong demo is honest about what remains', '09 · Limits & next steps', 10);
  const cols = [
    ['BUILT & RECORDED', [
      'Shared 15-fit accounting across rounds',
      'Fifteen saved CPU runs with artifact lineage',
      'Dashboard, ledger, and reproducible splits',
    ]],
    ['STILL TO VERIFY', [
      'Organizer data equivalence',
      'Official submission schema and hidden score',
      'Native GPU / tree-model round trips',
    ]],
    ['NEXT', [
      'Confirm roster and portal dataset',
      'Validate export against sample submission',
      'Verify organizer dataset and submission contract',
    ]],
  ];
  cols.forEach((col, i) => {
    const x = 0.68 + i * 3.02;
    box(pres, s, x, 1.57, 2.72, 2.85, i === 1 ? C.navy : C.blue, i === 1 ? C.gold : C.blue);
    text(s, col[0], x + 0.18, 1.83, 2.36, 0.24,
      { fontSize: 9, bold: true, color: C.gold, align: 'center', charSpacing: 0.5 });
    col[1].forEach((item, j) => {
      const y = 2.34 + j * 0.61;
      dot(pres, s, x + 0.2, y + 0.06, 0.1, i === 1 ? C.yellow : C.gold);
      text(s, item, x + 0.4, y, 2.1, 0.41,
        { fontSize: 9.1, color: C.yellow, valign: 'top' });
    });
  });
  text(s, 'Fifteen local fits complete the allocation — they are not proof of organizer performance.',
    0.8, 4.67, 8.4, 0.26, { fontSize: 12, bold: true, color: C.yellow, align: 'center' });
  return s;
}

const renderers = [null, page1, page2, page3, page4, page5, page6, page7, page8, page9, page10];
function render(pres, theme, index) {
  // Required theme contract is consumed by compile.js; the project palette is fixed here.
  void theme;
  return renderers[index](pres);
}
function preview(index) {
  const pres = new pptxgen();
  pres.defineLayout({ name: 'AUTONOMIS_WIDE', width: W, height: H });
  pres.layout = 'AUTONOMIS_WIDE';
  render(pres, { primary: C.ink, secondary: C.navy, accent: C.blue, light: C.gold, bg: C.ink }, index);
  return pres.writeFile({ fileName: `./output/slide-${String(index).padStart(2, '0')}-preview.pptx` });
}

module.exports = { render, preview, colors: C, width: W, height: H };
