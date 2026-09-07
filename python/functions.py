"""Shared helpers for the ADME notebook and port scripts."""
import os
from functools import lru_cache
from glob import glob
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm

IKCACHE = 'autoresearch/predict_adme/_ikcache'   # compound -> _ik parquets, one per public/internal source


def smiles_to_inchikeys(smiles, n_jobs=32, chunk=2000, v=True):
    """Parallel SMILES -> InChIKey with a progress bar.

    Batches the RDKit parse + InChIKey over n_jobs processes and shows one tqdm bar over completed
    batches. Much faster than a per-row comprehension for large sets (e.g. the ~330k MDCK frame).

    :param iterable smiles: SMILES strings
    :param int n_jobs: worker processes
    :param int chunk: SMILES per batch (one tqdm tick per batch)
    :param bool v: show the progress bar
    :return list: InChIKeys aligned to `smiles`; None for an unparseable/non-string SMILES
    """
    smiles = list(smiles)

    def _batch(batch):
        out = []
        for s in batch:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            out.append(Chem.MolToInchiKey(m) if m else None)
        return out

    # split into batches, run in parallel, keep submission order (return_as='generator')
    batches = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    gen = Parallel(n_jobs=n_jobs, return_as='generator')(delayed(_batch)(b) for b in batches)
    res = list(tqdm(gen, total=len(batches), disable=not v, desc='InChIKey'))
    return [ik for r in res for ik in r]


@lru_cache(maxsize=8)
def _load_ikcache(ikcache, id_col):
    """Read every <source>.parquet in the cache directory into one compound -> InChIKey dict (cached)."""
    ikmap = {}
    for f in sorted(glob(os.path.join(ikcache, '*.parquet'))):
        c = pd.read_parquet(f, columns=[id_col, '_ik'])
        ikmap.update(zip(c[id_col], c['_ik']))
    return ikmap


def inchikeys_for(df, id_col='compound', smiles_col='smiles', level='exact', ikcache=IKCACHE, v=False):
    """InChIKeys for a frame, from the `_ikcache` parquets first and RDKit only for what they miss.

    :param dataframe df: must contain id_col and smiles_col
    :param str level: 'exact' = the full 27-char key; 'skeleton' = block 1 only (14 chars), which also
        matches salts, stereoisomers, tautomers and charge states of the same connectivity
    :param str ikcache: directory of <source>.parquet files with columns [compound, _ik]
    :return series: keys aligned to df.index; an unparseable SMILES stays NaN
    """
    # one compound -> _ik lookup across every cached source (built once per directory)
    ikmap = _load_ikcache(ikcache, id_col)
    ik = pd.Series([ikmap.get(c) for c in df[id_col]], index=df.index, dtype=object)
    # RDKit fallback for the compounds the cache does not carry
    if ik.isna().any():
        ik.loc[ik.isna()] = smiles_to_inchikeys(df.loc[ik.isna(), smiles_col], v=v)
    return ik.str[:14] if level == 'skeleton' else ik


def drop_internal_twins(pub, internal, level='exact', id_col='compound', smiles_col='smiles',
                        ikcache=IKCACHE, v=True):
    """Remove every public row that is the SAME MOLECULE as an internal compound (leak control).

    A public row whose InChIKey matches an internal one must not pretrain a transfer model: the frozen
    encoder AND the loaded FFN head can carry a memorized molecule->value mapping into the finetune, even
    though no internal label ever enters the pretrain. Mirrors what `DATA.get_internal_public_sets` does
    for the RF augmented arms. `level='none'` disables the filter and returns `pub` unchanged.

    :param dataframe pub: public frame (the pretrain pool)
    :param dataframe internal: internal frame (every compound that can appear in a validation fold)
    :param str level: 'exact' | 'skeleton' | 'none'; see `inchikeys_for`
    :param bool v: print the dropped count
    :return tuple: (pub without the twins, boolean mask of the dropped rows on the ORIGINAL pub index)
    """
    if level == 'none':
        return pub, pd.Series(False, index=pub.index)
    # match public against internal on the requested key, then drop the matches
    keys_int = set(inchikeys_for(internal, id_col, smiles_col, level, ikcache).dropna())
    leak = inchikeys_for(pub, id_col, smiles_col, level, ikcache).isin(keys_int)
    if v:
        print(f'> leak control ({level}): dropped {int(leak.sum())} of {len(pub)} public rows '
              f'sharing a molecule with the {len(internal)} internal compounds')
    return pub[~leak].reset_index(drop=True), leak
