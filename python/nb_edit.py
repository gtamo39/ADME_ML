"""
Edit notebook code WITHOUT destroying the outputs in the live notebook.

Claude must never read a notebook's saved outputs (they hold df.head() tables, structure
thumbnails and per-compound values), and must never clear the live notebook either, because
the user wants those outputs. This module gives the safe middle path:

  1. `checkout` copies <nb>.ipynb to <nb>.claude.ipynb and clears the COPY's outputs.
     The copy carries no outputs, so Claude can read and edit it freely.
  2. Claude edits cells in the copy.
  3. `apply` writes a cell's SOURCE back into the live notebook, matched by cell id.
     Every other cell keeps its outputs untouched. Only the edited cell loses its own
     outputs, which are stale the moment the source changes.

CLI
    python python/nb_edit.py checkout vignettes/FP_preds.ipynb
    python python/nb_edit.py cells    vignettes/FP_preds.claude.ipynb
    python python/nb_edit.py apply    vignettes/FP_preds.ipynb --cell 8b258e7a
    python python/nb_edit.py apply    vignettes/FP_preds.ipynb --all

CAUTION: `apply` rewrites the live notebook. Close it in Jupyter first, or the editor may
save its in-memory copy over the change.
"""
import argparse
import hashlib
import json
import os
import shutil

WORK_SUFFIX = '.claude.ipynb'
BASELINE_KEY = 'claude_checkout'      # per-cell source hashes recorded in the scratch copy's metadata


def _src_hash(cell):
    """Hash of a cell's source alone — outputs never enter it."""
    return hashlib.sha1(''.join(cell['source']).encode()).hexdigest()[:12]


def work_path(live_path):
    """Path of the scratch copy that belongs to a live notebook."""
    return live_path[:-len('.ipynb')] + WORK_SUFFIX if live_path.endswith('.ipynb') else live_path + WORK_SUFFIX


def _read(path):
    with open(path) as fh:
        return json.load(fh)


def _write(nb, path):
    """Write a notebook back, keeping the indent nbformat uses so diffs stay small."""
    with open(path, 'w') as fh:
        json.dump(nb, fh, indent=1)
        fh.write('\n')


def _clear(cell):
    """Drop a code cell's outputs and reset its execution counter."""
    if cell['cell_type'] == 'code':
        cell['outputs'], cell['execution_count'] = [], None


def checkout(live_path, force=False):
    """Copy the live notebook to <nb>.claude.ipynb and clear every output in the copy.

    :param str live_path: the notebook the user works in
    :param bool force: overwrite an existing scratch copy
    :return str: path of the scratch copy
    """
    out = work_path(live_path)
    if os.path.exists(out) and not force:
        raise SystemExit(f'{out} already exists — pass --force to refresh it from the live notebook')
    shutil.copy2(live_path, out)
    nb = _read(out)
    for cell in nb['cells']:
        _clear(cell)
    # record what each cell's source looked like, so `apply` can detect a cell edited under us
    nb['metadata'][BASELINE_KEY] = {'live': live_path,
                                    'hashes': {c['id']: _src_hash(c) for c in nb['cells'] if c.get('id')}}
    _write(nb, out)
    n = sum(c['cell_type'] == 'code' for c in nb['cells'])
    print(f'> checkout: {live_path} -> {out} ({n} code cells, outputs cleared, baseline recorded)')
    return out


def cells(path):
    """Print each cell's index, id and first source line. Never touches outputs."""
    for i, c in enumerate(_read(path)['cells']):
        head = next((l.rstrip() for l in ''.join(c['source']).split('\n') if l.strip()), '(empty)')
        print('%3d  %-9s %-10s %s' % (i, c['cell_type'], c.get('id', '-'), head[:88]))


def apply_cells(live_path, from_path=None, cell_ids=(), all_changed=False, force=False):
    """Copy cell SOURCE from the scratch copy into the live notebook, matched by cell id.

    Only the cells named (or, with all_changed, every cell whose source differs) are touched.
    Each touched cell loses its own now-stale outputs; every other cell keeps its outputs.

    A cell whose live source changed since `checkout` is a genuine conflict — the user edited it
    while we held a copy — so the write is refused unless `force` is set. Cells we are NOT
    writing may change freely; the user can keep working elsewhere in the notebook.

    :param str live_path: the notebook the user works in
    :param str from_path: the scratch copy (default: work_path(live_path))
    :param tuple cell_ids: ids to copy across
    :param bool all_changed: copy every cell whose source differs
    :param bool force: write even where the live cell changed under us
    :return list: the ids written
    """
    from_path = from_path or work_path(live_path)
    live, work = _read(live_path), _read(from_path)
    by_id = {c.get('id'): c for c in live['cells'] if c.get('id')}
    baseline = work['metadata'].get(BASELINE_KEY, {}).get('hashes', {})

    if all_changed:
        cell_ids = [c['id'] for c in work['cells']
                    if c.get('id') in by_id and ''.join(c['source']) != ''.join(by_id[c['id']]['source'])]
    if not cell_ids:
        print('> apply: nothing to do (no cell ids given, or no source differs)')
        return []

    # refuse to clobber a cell the user edited in the live notebook since checkout
    conflicts = [cid for cid in cell_ids
                 if cid in baseline and cid in by_id and _src_hash(by_id[cid]) != baseline[cid]]
    if conflicts and not force:
        raise SystemExit(
            f'CONFLICT: {", ".join(conflicts)} changed in {live_path} since checkout.\n'
            f'The user edited those cells while we held a copy. Re-checkout and redo the edit on\n'
            f'their version, or pass --force to overwrite their change (asks for trouble).')

    written = []
    for cid in cell_ids:
        src = next((c for c in work['cells'] if c.get('id') == cid), None)
        if src is None:
            raise SystemExit(f'cell id {cid} is not in {from_path}')
        if cid not in by_id:
            raise SystemExit(f'cell id {cid} is not in {live_path}')
        dst = by_id[cid]
        dst['source'] = list(src['source'])
        # a cell can change kind (an empty markdown placeholder becoming code); keep the JSON legal
        if src['cell_type'] != dst['cell_type']:
            dst['cell_type'] = src['cell_type']
            if src['cell_type'] == 'code':
                dst.setdefault('outputs', []); dst.setdefault('execution_count', None)
            else:
                dst.pop('outputs', None); dst.pop('execution_count', None)
        _clear(dst)                      # the edited cell's own outputs are stale now
        written.append(cid)

    _write(live, live_path)
    kept = sum(1 for c in live['cells'] if c.get('outputs'))
    print(f'> apply: wrote {len(written)} cell(s) into {live_path} ({", ".join(written)}); '
          f'{kept} other cell(s) kept their outputs')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('checkout', help='copy the live notebook and clear the copy')
    p.add_argument('notebook')
    p.add_argument('--force', action='store_true', help='refresh an existing scratch copy')

    p = sub.add_parser('cells', help='list a notebook cells (index, id, first line)')
    p.add_argument('notebook')

    p = sub.add_parser('apply', help='copy edited cell source into the live notebook')
    p.add_argument('notebook')
    p.add_argument('--from', dest='from_path', default=None, help='scratch copy (default <nb>.claude.ipynb)')
    p.add_argument('--cell', action='append', default=[], help='cell id to copy (repeatable)')
    p.add_argument('--all', action='store_true', help='copy every cell whose source differs')
    p.add_argument('--force', action='store_true', help='write even over a cell the user changed since checkout')

    a = ap.parse_args()
    if a.cmd == 'checkout':
        checkout(a.notebook, a.force)
    elif a.cmd == 'cells':
        cells(a.notebook)
    else:
        apply_cells(a.notebook, a.from_path, tuple(a.cell), a.all, a.force)


if __name__ == '__main__':
    main()
