// app.js — drag-drop upload + LiveDesign-style grid render. Localhost only; no data leaves the box.
'use strict';

const $ = (id) => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), grid = $('grid'),
      statusEl = $('status'), legend = $('legend'), dlBtn = $('download'), clearBtn = $('clear');

let MODELS = null;   // {endpoints:[{key,unit,cutoff,favorable}], palette:{...}}

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

// compact numeric formatting across the wide ADME range (µM in the thousands down to logD ~1).
function fmt(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

// ---- header -----------------------------------------------------------------
async function loadModels() {
  MODELS = await (await fetch('/api/models')).json();
  const cut = (e) => e.cutoff == null ? '' : `${e.favorable || ''} ${e.cutoff}`;
  const th = ['<th>Structure</th>', '<th>Compound</th>', '<th>Score</th>'].concat(
    MODELS.endpoints.map((e) =>
      `<th>${e.key}<span class="u">${e.unit || ''}${cut(e) ? ' · ' + cut(e) : ''}</span></th>`));
  grid.innerHTML = `<thead><tr>${th.join('')}</tr></thead><tbody id="rows"></tbody>`;
}

// ---- cell + row render ------------------------------------------------------
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

function renderRows(rows) {
  const pal = MODELS.palette;
  $('rows').innerHTML = rows.map((r) => {
    const struct = r.valid && r.svg ? `<div class="struct">${r.svg}</div>` : '<div class="noparse">unparsed SMILES</div>';
    const cells = MODELS.endpoints.map((e) => epCell(r.preds[e.key], pal)).join('');
    return `<tr>
        <td class="meta struct">${struct}</td>
        <td class="meta cid">${r.compound == null ? '' : String(r.compound)}</td>
        <td class="meta"><input class="score" type="number" step="any" placeholder="—"></td>
        ${cells}
      </tr>`;
  }).join('');
}

// ---- upload -----------------------------------------------------------------
async function upload(files) {
  if (!files || !files.length) return;
  statusEl.textContent = `scoring ${files.length} file(s)…`;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) { statusEl.textContent = 'error: ' + (await res.text()).slice(0, 200); return; }
    const data = await res.json();
    renderRows(data.rows);
    legend.style.display = data.rows.length ? 'flex' : 'none';
    dlBtn.disabled = !data.rows.length;
    statusEl.textContent = `${data.n} compounds · ${data.n_valid} scored · ${data.n - data.n_valid} unparsed`;
  } catch (e) { statusEl.textContent = 'error: ' + e; }
}

// ---- events -----------------------------------------------------------------
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => upload(fileInput.files));
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', (e) => upload(e.dataTransfer.files));

dlBtn.addEventListener('click', () => { window.location = '/api/download'; });
clearBtn.addEventListener('click', () => {
  $('rows').innerHTML = ''; legend.style.display = 'none'; dlBtn.disabled = true;
  fileInput.value = ''; statusEl.textContent = 'idle';
});

loadModels();
