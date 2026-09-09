// app.js — drag-drop upload + LiveDesign-style grid render. Localhost only; no data leaves the box.
'use strict';

const $ = (id) => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), grid = $('grid'),
      statusEl = $('status'), legend = $('legend'), dlBtn = $('download'), clearBtn = $('clear'),
      mpoInput = $('mpo'), mpoStatus = $('mpostatus'),
      filterRowsEl = $('filterrows'), filterStatus = $('filterstatus');

let MODELS = null;         // {endpoints:[{key,unit,cutoff,favorable,transform}], palette:{...}}
let ROWS = [];             // every prediction row (kept client-side for sort + MPO retune)
let VISIBLE = [];          // the subset of ROWS the filter terms let through (what the grid shows)
let SORT = { key: null, dir: 1 };   // active sort column + direction (1 asc, -1 desc)
let FILTERS = [];          // filter terms [{field, op, value}] — an incomplete term is ignored
let FILTER_MODE = 'all';   // 'all' = every term must pass (AND), 'any' = at least one (OR)
let HIDDEN = new Set();    // column keys the user hid (still exported to the CSV)
let PAL = null;            // live cell palette; starts as the SERAC defaults from /api/models
let SHARP = 1;             // multiplies each column's color_slope (low = smoother value fade)
let BLEND = 10;            // half-width (%) of the gradient band across the cell split (0 = hard edge)
let SHAPE = 'curved';      // 'curved' (arc) or 'straight' (corner-to-corner diagonal)

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

// where a predicted value sits on the favorable axis: 0 = clearly bad, 0.5 = exactly at the
// triage cutoff, 1 = clearly good. Sigmoid of the distance to the cutoff in MODELLING space
// (so log endpoints fade per log unit), steered by the column's color_slope x the user's sharpness.
function gradPos(col, v) {
  if (v == null || col.cutoff == null || !col.favorable) return null;
  const x = toModel(col.transform, v), cut = toModel(col.transform, col.cutoff);
  if (!isFinite(x) || !isFinite(cut)) return null;
  const sign = (col.favorable === '>=' || col.favorable === '>') ? 1 : -1;
  const t = 1 / (1 + Math.exp(-sign * (col.color_slope == null ? 2 : col.color_slope) * SHARP * (x - cut)));
  return isFinite(t) ? t : null;
}

// predicted value -> diverging fade: pale at the cutoff, saturating toward olive (good) or red (bad).
function predColor(t, pal) {
  if (t == null) return '#eee8db';
  const [r, g, b] = hexToRgb(t >= 0.5 ? pal.fav : pal.unfav);
  const a = 0.10 + 0.80 * Math.min(1, Math.abs(t - 0.5) * 2);   // 0 at the cutoff -> 1 far from it
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
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

// ---- filters (client-side; hides rows, never re-scores them) -----------------
const OPS = { '>': (a, b) => a > b, '>=': (a, b) => a >= b, '<': (a, b) => a < b,
              '<=': (a, b) => a <= b, '=': (a, b) => a === b, '\u2260': (a, b) => a !== b };

const newTerm = () => ({ field: '', op: '>', value: '' });

// filterable properties: MPO, the manual Score, every column's value, and every column's confidence.
function filterFields() {
  const cols = MODELS.columns;
  return [{ v: 'mpo', t: 'MPO' }, { v: 'score', t: 'Score' },
          ...cols.map((e) => ({ v: e.key, t: (e.label || e.key) + (e.unit ? ` (${e.unit})` : '') })),
          ...cols.map((e) => ({ v: e.key + '_conf', t: e.key + '_conf (confidence)' }))];
}

// numeric value of one filter field on a row; null when the column is missing or the SMILES failed.
function fieldValue(r, f) {
  if (f === 'mpo') return r.mpo;
  if (f === 'score') return r.score;
  const p = r.preds[MODELS._byCol[f] ? f : f.replace(/_conf$/, '')];
  if (!p) return null;
  return MODELS._byCol[f] ? p.value : p.confidence;
}

// terms complete enough to apply (a property, a known operator and a numeric value).
const activeTerms = () => FILTERS.filter((t) => t.field && OPS[t.op] && t.value !== '' && isFinite(parseFloat(t.value)));

// a row passes when it satisfies every term ('all') or at least one term ('any'); a missing value fails.
function passRow(r, terms) {
  if (!terms.length) return true;
  const test = (t) => {
    const v = fieldValue(r, t.field);
    return (v == null || !isFinite(v)) ? false : OPS[t.op](v, parseFloat(t.value));
  };
  return FILTER_MODE === 'any' ? terms.some(test) : terms.every(test);
}

// report the term count and how many rows survive, in the (collapsible) panel header.
function updateFilterStatus(nTerms, nVis) {
  if (!nTerms) {
    filterStatus.textContent = ROWS.length ? `no filter · ${ROWS.length} rows` : '';
    filterStatus.className = 'filterstatus';
    return;
  }
  filterStatus.textContent = `${nTerms} term${nTerms > 1 ? 's' : ''} · ${nVis} of ${ROWS.length} rows`;
  filterStatus.className = 'filterstatus on';
}

// draw one row per term and wire its widgets; rebuilt only on add/remove (so typing keeps the focus).
let _ft = null;
function renderFilters() {
  const opts = filterFields();
  filterRowsEl.innerHTML = FILTERS.map((t, i) => `
      <div class="term">
        <select class="fld" data-i="${i}">
          <option value="">(select a property)</option>
          ${opts.map((o) => `<option value="${o.v}"${o.v === t.field ? ' selected' : ''}>${o.t}</option>`).join('')}
        </select>
        <select class="op" data-i="${i}">
          ${Object.keys(OPS).map((o) => `<option${o === t.op ? ' selected' : ''}>${o}</option>`).join('')}
        </select>
        <input class="val" data-i="${i}" type="number" step="any" placeholder="value" value="${t.value}">
        <button class="lnk rm" data-i="${i}">⊖ Remove term</button>
      </div>`).join('');
  filterRowsEl.querySelectorAll('select.fld').forEach((el) => el.addEventListener('change', () => {
    FILTERS[+el.dataset.i].field = el.value; renderRows();
  }));
  filterRowsEl.querySelectorAll('select.op').forEach((el) => el.addEventListener('change', () => {
    FILTERS[+el.dataset.i].op = el.value; renderRows();
  }));
  // debounce the value box so a fast typist does not re-render the grid on every keystroke
  filterRowsEl.querySelectorAll('input.val').forEach((el) => el.addEventListener('input', () => {
    FILTERS[+el.dataset.i].value = el.value;
    el.classList.toggle('err', el.value !== '' && !isFinite(parseFloat(el.value)));
    clearTimeout(_ft); _ft = setTimeout(renderRows, 150);
  }));
  // removing the last term leaves one empty row, so the builder always shows a term
  filterRowsEl.querySelectorAll('button.rm').forEach((el) => el.addEventListener('click', () => {
    FILTERS.splice(+el.dataset.i, 1);
    if (!FILTERS.length) FILTERS.push(newTerm());
    renderFilters(); renderRows();
  }));
}

// ---- columns panel: show / hide grid columns (a hidden column is still scored + exported) ----
function renderColumns() {
  const cols = [...META_COLS, ...MODELS.columns.map((c) => ({ k: c.key, t: c.label || c.key }))];
  $('colrows').innerHTML = cols.map((c) =>
    `<label class="colchk"><input type="checkbox" data-k="${c.k}"${shown(c.k) ? ' checked' : ''}>${c.t}</label>`).join('');
  $('colrows').querySelectorAll('input').forEach((el) => el.addEventListener('change', () => {
    if (el.checked) HIDDEN.delete(el.dataset.k); else HIDDEN.add(el.dataset.k);
    updateColStatus(); renderRows();
  }));
  updateColStatus();
}

// report how many columns are hidden, in the panel header.
function updateColStatus() {
  const n = HIDDEN.size;
  $('colstatus').textContent = n ? `${n} hidden` : '';
  $('colstatus').className = n ? 'filterstatus on' : 'filterstatus';
}

// ---- colors panel: override the cell palette (defaults = the SERAC config colors) ----
const COLOR_KEYS = ['fav', 'unfav', 'azure', 'ember'];

// push the live palette into the pickers, the sharpness readout, the legend swatches and its gradient bar.
function renderColors() {
  COLOR_KEYS.forEach((k) => { $('c_' + k).value = PAL[k]; });
  ['azure', 'ember'].forEach((k) => { $('l_' + k).style.background = PAL[k]; });
  $('c_sharp').value = SHARP;
  $('sharpval').textContent = SHARP.toFixed(2).replace(/0$/, '');
  // the cells read the split shape and the blend from CSS, so a change needs no row re-render
  $('c_blend').value = BLEND;
  $('c_shape').value = SHAPE;
  $('blendval').textContent = BLEND + '%';
  const root = document.documentElement;
  root.classList.toggle('shape-straight', SHAPE !== 'curved');   // curved is the CSS default
  root.style.setProperty('--blend', BLEND + '%');
  root.style.setProperty('--divop', Math.max(0, 1 - BLEND / 12).toFixed(2));
  // legend bar: the same fade the value triangles use, unfavorable -> cutoff -> favorable
  const fade = (hex, a) => { const [r, g, b] = hexToRgb(hex); return `rgba(${r},${g},${b},${a})`; };
  $('l_bar').style.background = `linear-gradient(to right, ${fade(PAL.unfav, 0.9)}, ${fade(PAL.unfav, 0.1)},` +
                                ` ${fade(PAL.fav, 0.1)}, ${fade(PAL.fav, 0.9)})`;
  const changed = COLOR_KEYS.filter((k) => PAL[k] !== MODELS.palette[k]).length
                  + (SHARP !== 1 ? 1 : 0) + (BLEND !== 10 ? 1 : 0) + (SHAPE !== 'curved' ? 1 : 0);
  $('colorstatus').textContent = changed ? `${changed} changed` : '';
  $('colorstatus').className = changed ? 'filterstatus on' : 'filterstatus';
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
  MODELS.extras = MODELS.extras || [];
  // grid columns = the 8 ADME endpoints + any extra model column; only endpoints feed the MPO scope.
  MODELS.columns = [...MODELS.endpoints, ...MODELS.extras];
  MODELS._byKey = Object.fromEntries(MODELS.endpoints.map((e) => [e.key, e]));
  MODELS._byCol = Object.fromEntries(MODELS.columns.map((e) => [e.key, e]));
  // <input type=color> only accepts lowercase #rrggbb, and the config palette is uppercase
  COLOR_KEYS.forEach((k) => { MODELS.palette[k] = String(MODELS.palette[k]).toLowerCase(); });
  PAL = { ...MODELS.palette };
  grid.innerHTML = '<thead></thead><tbody id="rows"></tbody>';
  renderHeader();
}

// is a column currently shown?
const shown = (k) => !HIDDEN.has(k);

// meta columns that can be hidden, in grid order (the select box is always shown).
const META_COLS = [{ k: 'struct', t: 'Structure' }, { k: 'compound', t: 'Compound' },
                   { k: 'score', t: 'Score' }, { k: 'mpo', t: 'MPO' }];

// rebuild the header row: sort arrows + the currently shown columns.
function renderHeader() {
  const cut = (e) => e.cutoff == null ? '' : `${e.favorable || ''} ${e.cutoff}`;
  const arrow = (k) => `<span class="ar">${SORT.key === k ? (SORT.dir > 0 ? '▲' : '▼') : ''}</span>`;
  const th = [`<th class="selcol"><input type="checkbox" id="selall" checked title="select all for download"></th>`];
  if (shown('struct')) th.push('<th>Structure</th>');
  if (shown('compound')) th.push(`<th data-k="compound" class="sortable">Compound${arrow('compound')}</th>`);
  if (shown('score')) th.push(`<th data-k="score" class="sortable">Score${arrow('score')}</th>`);
  if (shown('mpo')) th.push(`<th data-k="mpo" class="sortable">MPO${arrow('mpo')}</th>`);
  for (const e of MODELS.columns) {
    if (!shown(e.key)) continue;
    th.push(`<th data-k="${e.key}" class="sortable">${e.label || e.key}${arrow(e.key)}` +
            `<span class="u">${e.unit || ''}${cut(e) ? ' · ' + cut(e) : ''}</span></th>`);
  }
  grid.querySelector('thead').innerHTML = `<tr>${th.join('')}</tr>`;
  grid.querySelectorAll('th.sortable').forEach((el) => el.addEventListener('click', () => onSort(el.dataset.k)));
  // select-all toggles the download selection of the rows the filter shows
  $('selall').addEventListener('change', (e) => { VISIBLE.forEach((r) => r.selected = e.target.checked); renderRows(); });
}

// ---- cell + row render -------------------------------------------------------
function epCell(p, pal, col) {
  if (!p || p.value == null) return '<td class="epcell empty"></td>';
  return `<td class="epcell">
      <div class="half conf" style="background:${confColor(p.confidence, pal)}"></div>
      <div class="half pred" style="background:${predColor(gradPos(col, p.value), pal)}"></div>
      <div class="divider"></div>
      <b class="cv">${p.confidence.toFixed(2)}</b>
      <b class="pv">${fmt(p.value)}</b>
    </td>`;
}

function renderRows() {
  const pal = PAL || MODELS.palette;
  renderHeader();
  // keep the ROWS index on every cell (data-i) so sort/filter never mis-target a row's state
  const terms = activeTerms();
  const vis = ROWS.map((r, i) => ({ r, i })).filter((x) => passRow(x.r, terms));
  VISIBLE = vis.map((x) => x.r);
  updateFilterStatus(terms.length, vis.length);
  $('rows').innerHTML = vis.map(({ r, i }) => {
    const struct = r.valid && r.svg ? `<div class="struct">${r.svg}</div>` : '<div class="noparse">unparsed SMILES</div>';
    const cells = MODELS.columns.filter((e) => shown(e.key)).map((e) => epCell(r.preds[e.key], pal, e)).join('');
    const score = `<input class="score" data-i="${i}" type="number" step="any" placeholder="—" value="${r.score == null ? '' : r.score}">`;
    const td = [`<td class="meta sel"><input type="checkbox" class="rowsel" data-i="${i}" ${r.selected ? 'checked' : ''}></td>`];
    if (shown('struct')) td.push(`<td class="meta struct" data-i="${i}">${struct}</td>`);
    if (shown('compound')) td.push(`<td class="meta cid">${r.compound == null ? '' : String(r.compound)}</td>`);
    if (shown('score')) td.push(`<td class="meta">${score}</td>`);
    if (shown('mpo')) td.push(`<td class="mpocell" style="background:${mpoColor(r.mpo, pal)}">${r.mpo == null ? '—' : r.mpo.toFixed(3)}</td>`);
    return `<tr>${td.join('')}${cells}</tr>`;
  }).join('');
  // keep the manual Score edits in row state so they survive a re-sort / re-render.
  $('rows').querySelectorAll('input.score').forEach((el) => el.addEventListener('input', () => {
    ROWS[+el.dataset.i].score = el.value === '' ? null : parseFloat(el.value);
  }));
  // per-row download selection
  $('rows').querySelectorAll('input.rowsel').forEach((el) => el.addEventListener('change', () => {
    ROWS[+el.dataset.i].selected = el.checked; updateSelAll();
  }));
  updateSelAll();
}

// reflect the VISIBLE row selection in the header select-all box (checked / unchecked / indeterminate).
function updateSelAll() {
  const sa = $('selall');
  if (!sa || !VISIBLE.length) return;
  const on = VISIBLE.filter((r) => r.selected).length;
  sa.checked = on === VISIBLE.length;
  sa.indeterminate = on > 0 && on < VISIBLE.length;
}

// ---- upload ------------------------------------------------------------------
// write the status line
function setStatus(text) { statusEl.textContent = text; }

// show/advance/hide the scoring progress bar. /api/predict streams one line per finished stage, so
// this is real progress, not a guess — the chemprop stages take seconds each (one CLI subprocess).
function setProgress(done, total) {
  const on = done != null;
  $('prog').classList.toggle('on', on);
  if (!on) return;
  const pct = total ? Math.round((100 * done) / total) : 0;
  $('progfill').style.width = pct + '%';
  $('progpct').textContent = pct + '%';
}

// read a newline-delimited-JSON body line by line, handing each parsed object to onLine
async function readNdjson(res, onLine) {
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    buf += done ? '' : dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onLine(JSON.parse(line));
    }
    if (done) break;
  }
}

async function upload(files) {
  if (!files || !files.length) return;
  setStatus(`scoring ${files.length} file(s)…`);
  setProgress(0, 1);
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) { setProgress(null); setStatus('error: ' + (await res.text()).slice(0, 200)); return; }
    let data = null, err = null;
    await readNdjson(res, (m) => {
      if (m.result) data = m.result;
      else if (m.error) err = m.error;
      else { setProgress(m.done, m.total); setStatus(`scoring · ${m.label}`); }
    });
    setProgress(null);
    if (err) { setStatus('error: ' + err); return; }
    if (!data) { setStatus('error: the prediction stream ended with no result'); return; }
    ROWS = data.rows.map((r) => ({ ...r, score: null, mpo: null, selected: true }));
    recomputeMPO();
    if (SORT.key) sortRows();
    renderRows();
    legend.style.display = ROWS.length ? 'flex' : 'none';
    dlBtn.disabled = !ROWS.length;
    setStatus(`${data.n} compounds · ${data.n_valid} scored · ${data.n - data.n_valid} unparsed`);
  } catch (e) { setProgress(null); setStatus('error: ' + e); }
}

// ---- hover: high-resolution structure preview -------------------------------
const preview = $('preview');
// position the floating preview near the cursor, clamped to the viewport
function placePreview(e) {
  const pad = 16, w = preview.offsetWidth || 380, h = preview.offsetHeight || 300;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth) x = e.clientX - w - pad;
  if (y + h > innerHeight) y = innerHeight - h - pad;
  preview.style.left = Math.max(pad, x) + 'px';
  preview.style.top = Math.max(pad, y) + 'px';
}
// show a FRESH high-resolution render (thin, clean bonds) on hover over a structure cell; hide on leave
grid.addEventListener('mouseover', (e) => {
  const cell = e.target.closest('td.struct');
  if (!cell || cell.dataset.i == null) return;
  const r = ROWS[+cell.dataset.i];
  const html = r && (r.svg_hi || r.svg);      // hi-res preview, fall back to the thumbnail
  if (!html) return;
  preview.innerHTML = html; preview.style.display = 'block'; placePreview(e);
});
grid.addEventListener('mousemove', (e) => { if (preview.style.display === 'block') placePreview(e); });
grid.addEventListener('mouseout', (e) => {
  const cell = e.target.closest('td.struct');
  if (cell && !cell.contains(e.relatedTarget)) preview.style.display = 'none';
});

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

// build a CSV client-side from the SELECTED rows (respects the checkboxes; adds MPO + Score).
function toCSV(rows) {
  const eps = MODELS.columns.map((e) => e.key);      // hidden columns are exported too
  const head = ['compound', 'smiles', 'mpo', 'score', ...eps.flatMap((k) => [k + '_pred', k + '_confidence'])];
  const esc = (v) => { if (v == null) return ''; const s = String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const lines = [head.join(',')];
  for (const r of rows) {
    const row = [r.compound, r.smiles, r.mpo == null ? '' : r.mpo.toFixed(4), r.score == null ? '' : r.score];
    for (const k of eps) { const p = r.preds[k] || {}; row.push(p.value == null ? '' : p.value, p.confidence == null ? '' : p.confidence); }
    lines.push(row.map(esc).join(','));
  }
  return lines.join('\n');
}

dlBtn.addEventListener('click', () => {
  const sel = VISIBLE.filter((r) => r.selected);      // filtered-out rows are never exported
  if (!sel.length) { setStatus('no compounds selected'); return; }
  // trigger a local download of the selected rows (nothing leaves the machine)
  const url = URL.createObjectURL(new Blob([toCSV(sel)], { type: 'text/csv' }));
  const a = document.createElement('a'); a.href = url; a.download = 'adme_predictions.csv'; a.click();
  URL.revokeObjectURL(url);
  setStatus(`downloaded ${sel.length} of ${VISIBLE.length} shown compounds`);
});
clearBtn.addEventListener('click', () => {
  ROWS = []; VISIBLE = []; SORT = { key: null, dir: 1 };
  renderRows(); legend.style.display = 'none'; dlBtn.disabled = true;
  preview.style.display = 'none';
  const sa = $('selall'); if (sa) { sa.checked = true; sa.indeterminate = false; }
  fileInput.value = ''; setStatus('idle');
});

// filter panel: add / clear terms and switch the AND-OR mode; every change re-renders the grid.
$('addterm').addEventListener('click', () => { FILTERS.push(newTerm()); renderFilters(); });
$('clearterms').addEventListener('click', () => { FILTERS = [newTerm()]; renderFilters(); renderRows(); });
$('filtermode').addEventListener('change', (e) => { FILTER_MODE = e.target.value; renderRows(); });

// columns panel: show every column, or hide every model column at once.
$('showallcols').addEventListener('click', () => { HIDDEN.clear(); renderColumns(); renderRows(); });
$('hideallcols').addEventListener('click', () => {
  MODELS.columns.forEach((c) => HIDDEN.add(c.key)); renderColumns(); renderRows();
});

// colors panel: a picker retunes the palette live; reset restores the SERAC config colors.
COLOR_KEYS.forEach((k) => $('c_' + k).addEventListener('input', (e) => {
  PAL[k] = e.target.value; renderColors(); renderRows();
}));
$('c_sharp').addEventListener('input', (e) => { SHARP = parseFloat(e.target.value); renderColors(); renderRows(); });
// the split shape and blend live in CSS, so they need renderColors only — no row rebuild
$('c_blend').addEventListener('input', (e) => { BLEND = parseFloat(e.target.value); renderColors(); });
$('c_shape').addEventListener('change', (e) => { SHAPE = e.target.value; renderColors(); });
$('resetcolors').addEventListener('click', () => {
  PAL = { ...MODELS.palette }; SHARP = 1; BLEND = 10; SHAPE = 'curved'; renderColors(); renderRows();
});

// default MPO = equal-weight mean of each endpoint's sigmoid-at-cutoff desirability.
async function init() {
  await loadModels();
  mpoInput.value = 'mean(' + MODELS.endpoints.map((e) =>
    e.cutoff == null ? `sigmoid('${e.key}')` : `sigmoid('${e.key}', ${e.cutoff})`).join(', ') + ')';
  // start with one empty term, like the reference builder
  FILTERS = [newTerm()];
  renderFilters(); renderColumns(); renderColors();
}
init();
