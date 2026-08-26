// app.js — drag-drop upload + LiveDesign-style grid render. Localhost only; no data leaves the box.
'use strict';

const $ = (id) => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), grid = $('grid'),
      statusEl = $('status'), legend = $('legend'), dlBtn = $('download'), clearBtn = $('clear'),
      mpoInput = $('mpo'), mpoStatus = $('mpostatus');

let MODELS = null;         // {endpoints:[{key,unit,cutoff,favorable,transform}], palette:{...}}
let ROWS = [];             // current prediction rows (kept client-side for sort + MPO retune)
let SORT = { key: null, dir: 1 };   // active sort column + direction (1 asc, -1 desc)

// ---- color helpers ----------------------------------------------------------
const hexToRgb = (h) => { const n = parseInt(h.replace('#', ''), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };

// confidence -> diverging gradient centered at split: azure above, ember below, paler near the split.
function confColor(c, pal) {
  if (c == null) return '#f6f3ec';
  const split = pal.split, base = c > split ? pal.azure : pal.ember;
  const t = c > split ? (c - split) / (1 - split) : (split - c) / split;   // 0 at split -> 1 at extreme
  const a = 0.18 + 0.82 * Math.min(1, Math.max(0, t));                       // alpha as the gradient
  const [r, g, b] = hexToRgb(base);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

// predicted value -> favorable (olive) / unfavorable (red) / unknown (grey).
function predColor(fav, pal) {
  if (fav === true) { const [r, g, b] = hexToRgb(pal.fav); return `rgba(${r},${g},${b},0.85)`; }
  if (fav === false) { const [r, g, b] = hexToRgb(pal.unfav); return `rgba(${r},${g},${b},0.85)`; }
  return '#eee8db';
}

// MPO score -> olive intensity by the value clamped to [0,1] (green = good, pale = poor).
function mpoColor(v, pal) {
  if (v == null || !isFinite(v)) return '#f6f3ec';
  const t = Math.min(1, Math.max(0, v));
  const [r, g, b] = hexToRgb(pal.fav);
  return `rgba(${r},${g},${b},${(0.12 + 0.78 * t).toFixed(3)})`;
}

// compact numeric formatting across the wide ADME range (µM in the thousands down to logD ~1).
function fmt(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

// ---- MPO expression evaluator (client-side; localhost single user) ----------

// per-endpoint transform into the modelling space where cutoffs live.
function toModel(tf, v) {
  if (v == null || !isFinite(v)) return NaN;
  if (tf === 'log10') return Math.log10(v);
  if (tf === 'logit_pct') return Math.log10(v / (100 - v));
  return v;                                   // identity
}

// desirability: sigmoid toward the favorable side, in transform space, [0,1].
// center defaults to the triage cutoff; the favorable inequality sets the direction.
function desirability(ep, center, slope) {
  const c = MODELS._byKey[ep];
  if (!c || c.favorable == null) return NaN;
  const cut = (center == null ? c.cutoff : center);   // 2nd arg overrides the cutoff (raw units)
  if (cut == null) return NaN;
  const sign = (c.favorable === '>=' || c.favorable === '>') ? 1 : -1;   // higher-better vs lower-better
  const s = (slope == null ? 2 : slope);
  return 1 / (1 + Math.exp(-sign * s * (toModel(c.transform, this._v[ep]) - toModel(c.transform, cut))));
}

// build the compiled MPO function from the formula string; throws on a bad expression.
function compileMPO(expr) {
  if (/=>|\bfunction\b|`|\bwindow\b|\bdocument\b|\bfetch\b|\bthis\b/.test(expr))
    throw new Error('disallowed token in formula');
  // scope exposes bare endpoint names (raw value), helpers, and Math primitives.
  const fn = new Function('S', 'with (S) { return (' + expr + '); }');
  return (row) => {
    const scope = { _v: {} };
    for (const e of MODELS.endpoints) {
      const p = row.preds[e.key];
      scope[e.key] = (p && p.value != null) ? p.value : NaN;   // bare name -> raw predicted value
      scope._v[e.key] = scope[e.key];
    }
    scope.d = desirability.bind(scope);                         // d('ep'[, center, slope]) -> desirability
    scope.c = (ep) => { const p = row.preds[ep]; return (p && p.confidence != null) ? p.confidence : NaN; };
    // sigmoid('ep'[, center, slope]) centers at that endpoint's cutoff (favorable-oriented, = d);
    // sigmoid(x, center, slope) is the manual numeric form.
    scope.sigmoid = (x, center, slope) => (typeof x === 'string')
      ? desirability.call(scope, x, center, slope)
      : 1 / (1 + Math.exp(-(slope == null ? 1 : slope) * (x - (center || 0))));
    scope.mean = (...a) => a.reduce((s, x) => s + x, 0) / a.length;
    scope.clamp = (x, lo, hi) => Math.min(hi == null ? 1 : hi, Math.max(lo == null ? 0 : lo, x));
    Object.assign(scope, { min: Math.min, max: Math.max, exp: Math.exp, log: Math.log, abs: Math.abs, pow: Math.pow });
    const val = fn(scope);
    return (typeof val === 'number' && isFinite(val)) ? val : null;
  };
}

// recompute the mpo field on every row from the current formula; report parse errors.
function recomputeMPO() {
  const expr = mpoInput.value.trim();
  if (!expr) { ROWS.forEach((r) => r.mpo = null); mpoStatus.textContent = ''; return; }
  let f;
  try { f = compileMPO(expr); } catch (e) { mpoStatus.textContent = '✗ ' + e.message; mpoStatus.className = 'mpostatus err'; return; }
  try { ROWS.forEach((r) => r.mpo = r.valid ? f(r) : null); }
  catch (e) { mpoStatus.textContent = '✗ ' + e.message; mpoStatus.className = 'mpostatus err'; return; }
  mpoStatus.textContent = '✓ applied'; mpoStatus.className = 'mpostatus ok';
}

// ---- sort --------------------------------------------------------------------

// value used to sort a row on the active column (string for meta, number for scores/endpoints).
function sortValue(r, key) {
  if (key === 'compound') return String(r.compound == null ? '' : r.compound).toLowerCase();
  if (key === 'score') return r.score == null ? NaN : r.score;
  if (key === 'mpo') return r.mpo == null ? NaN : r.mpo;
  const p = r.preds[key];
  return (p && p.value != null) ? p.value : NaN;
}

function sortRows() {
  if (!SORT.key) return;
  const k = SORT.key, dir = SORT.dir;
  ROWS.sort((a, b) => {
    const x = sortValue(a, k), y = sortValue(b, k);
    const xn = (typeof x === 'number'), nx = xn && isNaN(x), ny = xn && isNaN(y);
    if (nx && ny) return 0; if (nx) return 1; if (ny) return -1;   // blanks always sink to the bottom
    return x < y ? -dir : x > y ? dir : 0;
  });
}

function onSort(key) {
  SORT.dir = (SORT.key === key) ? -SORT.dir : (key === 'compound' ? 1 : -1);   // default numeric = high-first
  SORT.key = key;
  sortRows(); renderRows();
}

// ---- header ------------------------------------------------------------------
async function loadModels() {
  MODELS = await (await fetch('/api/models')).json();
  MODELS._byKey = Object.fromEntries(MODELS.endpoints.map((e) => [e.key, e]));
  const cut = (e) => e.cutoff == null ? '' : `${e.favorable || ''} ${e.cutoff}`;
  const arrow = (k) => `<span class="ar">${SORT.key === k ? (SORT.dir > 0 ? '▲' : '▼') : ''}</span>`;
  const th = [`<th data-k="compound">Structure</th>`,
              `<th data-k="compound" class="sortable">Compound${arrow('compound')}</th>`,
              `<th data-k="score" class="sortable">Score${arrow('score')}</th>`,
              `<th data-k="mpo" class="sortable">MPO${arrow('mpo')}</th>`].concat(
    MODELS.endpoints.map((e) =>
      `<th data-k="${e.key}" class="sortable">${e.key}${arrow(e.key)}<span class="u">${e.unit || ''}${cut(e) ? ' · ' + cut(e) : ''}</span></th>`));
  grid.innerHTML = `<thead><tr>${th.join('')}</tr></thead><tbody id="rows"></tbody>`;
  grid.querySelectorAll('th.sortable').forEach((th) => th.addEventListener('click', () => onSort(th.dataset.k)));
}

// ---- cell + row render -------------------------------------------------------
function epCell(p, pal) {
  if (!p || p.value == null) return '<td class="epcell empty"></td>';
  return `<td class="epcell">
      <div class="half conf" style="background:${confColor(p.confidence, pal)}"></div>
      <div class="half pred" style="background:${predColor(p.favorable, pal)}"></div>
      <div class="divider"></div>
      <b class="cv">${p.confidence.toFixed(2)}</b>
      <b class="pv">${fmt(p.value)}</b>
    </td>`;
}

function renderRows() {
  const pal = MODELS.palette;
  refreshHeader();
  $('rows').innerHTML = ROWS.map((r, i) => {
    const struct = r.valid && r.svg ? `<div class="struct">${r.svg}</div>` : '<div class="noparse">unparsed SMILES</div>';
    const cells = MODELS.endpoints.map((e) => epCell(r.preds[e.key], pal)).join('');
    const mpo = `<td class="mpocell" style="background:${mpoColor(r.mpo, pal)}">${r.mpo == null ? '—' : r.mpo.toFixed(3)}</td>`;
    const score = `<input class="score" data-i="${i}" type="number" step="any" placeholder="—" value="${r.score == null ? '' : r.score}">`;
    return `<tr>
        <td class="meta struct">${struct}</td>
        <td class="meta cid">${r.compound == null ? '' : String(r.compound)}</td>
        <td class="meta">${score}</td>
        ${mpo}
        ${cells}
      </tr>`;
  }).join('');
  // keep the manual Score edits in row state so they survive a re-sort / re-render.
  $('rows').querySelectorAll('input.score').forEach((el) => el.addEventListener('input', () => {
    ROWS[+el.dataset.i].score = el.value === '' ? null : parseFloat(el.value);
  }));
}

// update only the header sort arrows in place (avoids rebuilding the whole header).
function refreshHeader() {
  grid.querySelectorAll('th.sortable').forEach((th) => {
    const ar = th.querySelector('.ar');
    if (ar) ar.textContent = SORT.key === th.dataset.k ? (SORT.dir > 0 ? '▲' : '▼') : '';
  });
}

// ---- upload ------------------------------------------------------------------
async function upload(files) {
  if (!files || !files.length) return;
  statusEl.textContent = `scoring ${files.length} file(s)…`;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) { statusEl.textContent = 'error: ' + (await res.text()).slice(0, 200); return; }
    const data = await res.json();
    ROWS = data.rows.map((r) => ({ ...r, score: null, mpo: null }));
    recomputeMPO();
    if (SORT.key) sortRows();
    renderRows();
    legend.style.display = ROWS.length ? 'flex' : 'none';
    dlBtn.disabled = !ROWS.length;
    statusEl.textContent = `${data.n} compounds · ${data.n_valid} scored · ${data.n - data.n_valid} unparsed`;
  } catch (e) { statusEl.textContent = 'error: ' + e; }
}

// ---- events ------------------------------------------------------------------
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => upload(fileInput.files));
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', (e) => upload(e.dataTransfer.files));

// retune MPO live as the formula changes; re-sort in place when MPO is the active column.
let _t = null;
mpoInput.addEventListener('input', () => {
  clearTimeout(_t);
  _t = setTimeout(() => { recomputeMPO(); if (SORT.key === 'mpo') sortRows(); renderRows(); }, 180);
});

dlBtn.addEventListener('click', () => { window.location = '/api/download'; });
clearBtn.addEventListener('click', () => {
  ROWS = []; SORT = { key: null, dir: 1 };
  $('rows').innerHTML = ''; legend.style.display = 'none'; dlBtn.disabled = true;
  fileInput.value = ''; statusEl.textContent = 'idle';
});

// default MPO = equal-weight mean of each endpoint's sigmoid-at-cutoff desirability.
async function init() {
  await loadModels();
  mpoInput.value = 'mean(' + MODELS.endpoints.map((e) =>
    e.cutoff == null ? `sigmoid('${e.key}')` : `sigmoid('${e.key}', ${e.cutoff})`).join(', ') + ')';
}
init();
