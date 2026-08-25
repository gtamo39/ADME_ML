"""Shared helpers for the ADME notebook and port scripts."""
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm


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
