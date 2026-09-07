'''
Derive a 3Di structural sequence for every protein, from the AlphaFold DB models
fetched by `fetch_afdb_structures.py`.

3Di is Foldseek's structural alphabet (van Kempen et al. 2024, Nat Biotechnol
42:243): one letter per residue describing that residue's local tertiary
environment -- its nearest neighbour in 3D and the geometry between them. It is
what makes ProstT5 bilingual: the model was trained to translate between amino
acids and 3Di, so it will accept a 3Di string as input directly, with the
`<fold2AA>` prefix instead of `<AA2fold>`.

That is the whole point of using ProstT5 here rather than any other PLM. The
sequence-only arm asks ProstT5 for its opinion about a sequence. This asks it
about a structure as well, in the second language it was actually trained on.

WHY NOT LET ProstT5 PREDICT THE 3Di ITSELF
ProstT5 can translate AA -> 3Di. Doing that and feeding the result back would add
no information: it is a deterministic function of the sequence the model already
has. The point of the structure arm is to inject evidence the sequence-only arm
did not have, so the 3Di must come from a structure -- here an AlphaFold model,
which is itself predicted from sequence but by a different model, with an MSA,
and is therefore not recoverable from ProstT5's own forward pass.

TWO ENCODERS, ONE ALPHABET
    --encoder mini3di   (default) pure Python, `pip install mini3di`. A
                        reimplementation of Foldseek's 3Di encoder by Larralde,
                        using the same published VQ-VAE weights. No binary, no
                        conda, works on macOS and on a login node.
    --encoder foldseek  the reference implementation, via
                        `foldseek structureto3didescriptor`. Use it if you need
                        bit-identical agreement with Foldseek output.

They agree closely but not always bit-for-bit. Whichever is used is recorded in
manifest_3di.json, because a 3Di set is only comparable with another produced the
same way.

ALIGNMENT IS THE THING THAT CAN GO SILENTLY WRONG
The 3Di string must be the same length as the amino-acid sequence and in the same
order, or every downstream per-residue embedding is shifted against the labels
and the run still completes, reporting a plausible-looking F1. So:

  * the AFDB model's own sequence is compared to ours and must match EXACTLY.
    A near match is a different isoform; AFDB also fragments sequences over
    ~2700 aa into F1/F2/... files, and the F1 fragment covers only part of the
    protein. Both fail the comparison and are recorded as unusable rather than
    patched up.
  * the emitted 3Di length is checked against the sequence length before it is
    written.
  * proteins with no usable structure are KEPT, with `three_di: null`. They are
    masked at embedding time, not dropped -- dropping them would change the
    GraphPart partitions and break comparability with every sequence-only run.

Output: `three_di.json`, mapping the md5 of the amino-acid sequence to its 3Di
string, plus `manifest_3di.json` recording the encoder, the counts and the
per-accession status. Keying on the sequence hash matches
`make_embeddings.py`/`dataset.py`, so nothing downstream has to know about
accessions.

Usage
-----
    pip install mini3di biotite
    python -m src.utils.fetch_afdb_structures --out_dir structures/
    python -m src.utils.make_3di --structures_dir structures/ --out_dir three_di/
'''
import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from hashlib import md5

import pandas as pd
from tqdm.auto import tqdm


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).digest().hex()


def _backbone_arrays(pdb_path: str):
    '''Return (ca, cb, n, c) coordinate arrays and the AA sequence, or None.

    mini3di wants one row per residue for each of the four backbone atoms, with
    NaN where an atom is absent -- glycine has no CB, and that is expected rather
    than an error. Residue order comes from the sorted residue ids, so a PDB whose
    ATOM records are out of order still yields an in-order array.
    '''
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.info import one_letter_code
    from biotite.structure.io.pdb import PDBFile

    try:
        atoms = PDBFile.read(pdb_path).get_structure(model=1)
    except Exception:
        return None
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if atoms.array_length() == 0:
        return None

    residue_ids = np.unique(atoms.res_id)
    position_of = {res_id: i for i, res_id in enumerate(residue_ids)}
    n_res = len(residue_ids)

    coords = {name: np.full((n_res, 3), np.nan, dtype=np.float32)
              for name in ('CA', 'CB', 'N', 'C')}
    for atom in atoms:
        if atom.atom_name in coords:
            coords[atom.atom_name][position_of[atom.res_id]] = atom.coord

    starts = struc.get_residue_starts(atoms)
    sequence = ''.join(one_letter_code(name) or 'X' for name in atoms.res_name[starts])
    if len(sequence) != n_res:
        return None

    return coords['CA'], coords['CB'], coords['N'], coords['C'], sequence


def encode_mini3di(pdb_path: str, encoder):
    '''3Di string via mini3di. Returns (three_di, structure_sequence) or None.'''
    loaded = _backbone_arrays(pdb_path)
    if loaded is None:
        return None
    ca, cb, n, c, sequence = loaded
    try:
        states = encoder.encode_atoms(ca=ca, cb=cb, n=n, c=c)
        three_di = encoder.build_sequence(states)
    except Exception:
        return None
    if len(three_di) != len(sequence):
        return None
    return three_di, sequence


def encode_foldseek(pdb_path: str, binary: str):
    '''3Di string via the Foldseek binary. Returns (three_di, sequence) or None.

    `structureto3didescriptor` writes a TSV whose columns are
    name, amino-acid sequence, 3Di sequence, ... -- the first three are all we
    need, and taking them positionally avoids depending on the column count,
    which has changed between Foldseek releases.
    '''
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, 'out.tsv')
        try:
            subprocess.run([binary, 'structureto3didescriptor', pdb_path, out_path],
                           check=True, capture_output=True, timeout=600)
        except Exception:
            return None
        try:
            first = open(out_path).readline().rstrip('\n')
        except OSError:
            return None
    fields = first.split('\t')
    if len(fields) < 3:
        return None
    sequence, three_di = fields[1], fields[2]
    if not three_di or len(three_di) != len(sequence):
        return None
    return three_di, sequence


def generate(data_file: str, structures_dir: str, out_dir: str, encoder_name: str,
             foldseek_binary: str, limit: int) -> None:
    frame = pd.read_csv(data_file)

    manifest_path = os.path.join(structures_dir, 'manifest.json')
    if not os.path.isfile(manifest_path):
        raise SystemExit(
            f'No manifest at {manifest_path}. Run '
            '`python -m src.utils.fetch_afdb_structures --out_dir '
            f'{structures_dir}` first.')
    structures = json.load(open(manifest_path))
    n_ok = sum(1 for v in structures.values() if v.get('status') == 'ok')
    print(f'manifest: {n_ok}/{len(structures)} accessions have a usable structure')

    encoder = None
    if encoder_name == 'mini3di':
        try:
            import mini3di
        except ImportError:
            raise SystemExit('mini3di is not installed. `pip install mini3di biotite`, '
                             'or pass --encoder foldseek.')
        encoder = mini3di.Encoder()
    else:
        import shutil
        if shutil.which(foldseek_binary) is None:
            raise SystemExit(f'{foldseek_binary} is not on PATH. Install Foldseek, '
                             'or use the default --encoder mini3di.')

    # Deduplicate by sequence hash exactly as make_embeddings.py does, and prefer
    # an accession that HAS a structure. Taking the first accession blindly would
    # silently lose the structure for any sequence whose first accession is the
    # one AFDB happens to be missing.
    best = {}
    for accession, sequence in zip(frame['protein_id'].astype(str),
                                   frame['sequence'].astype(str)):
        digest = hash_aa_string(sequence)
        has_structure = structures.get(accession, {}).get('status') == 'ok'
        if digest not in best or (has_structure and not best[digest][1]):
            best[digest] = ((accession, sequence, digest), has_structure)
    records = [value[0] for value in best.values()]
    print(f'{len(records)} unique sequences; '
          f'{sum(1 for v in best.values() if v[1])} have a structure')

    if limit:
        records = records[:limit]
        print(f'--limit {limit}: smoke test only')

    three_di_by_hash = {}
    per_accession = {}
    stats = Counter()

    for accession, sequence, digest in tqdm(records):
        status = 'no_structure'
        three_di = None

        if structures.get(accession, {}).get('status') == 'ok':
            pdb_path = os.path.join(structures_dir, f'{accession}.pdb')
            if not os.path.isfile(pdb_path):
                status = 'pdb_missing'
            else:
                result = (encode_mini3di(pdb_path, encoder) if encoder is not None
                          else encode_foldseek(pdb_path, foldseek_binary))
                if result is None:
                    status = 'encode_failed'
                else:
                    candidate, structure_sequence = result
                    # Exact match only. A per-residue track aligned to the wrong
                    # residues is worse than no track at all, and every way of
                    # "fixing" a mismatch here (trimming, padding, aligning)
                    # invents an alignment the label positions do not share.
                    if structure_sequence != sequence:
                        status = 'sequence_mismatch'
                    elif len(candidate) != len(sequence):
                        status = 'length_mismatch'
                    else:
                        status = 'ok'
                        three_di = candidate

        stats[status] += 1
        per_accession[accession] = {'status': status, 'hash': digest,
                                    'length': len(sequence)}
        three_di_by_hash[digest] = three_di

    os.makedirs(out_dir, exist_ok=True)
    json.dump(three_di_by_hash, open(os.path.join(out_dir, 'three_di.json'), 'w'))
    json.dump(
        {'encoder': encoder_name,
         'foldseek_binary': foldseek_binary if encoder_name == 'foldseek' else None,
         'structures_dir': structures_dir, 'data_file': data_file,
         'counts': dict(stats), 'per_accession': per_accession},
        open(os.path.join(out_dir, 'manifest_3di.json'), 'w'), indent=1)

    print('\n=== 3Di summary ===')
    for status, count in stats.most_common():
        print(f'  {status:20} {count:6}')
    total = sum(stats.values())
    if total:
        print(f'\n  {100*stats["ok"]/total:.2f}% of sequences carry a real 3Di '
              f'string. The rest are masked at embedding time, not dropped.')
    if stats['ok'] == 0:
        print('\n  WARNING: nothing was encoded. The structure arm would be '
              'identical to the sequence-only baseline -- check the manifest.')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_file', default='data/labeled_sequences.csv')
    parser.add_argument('--structures_dir', required=True,
                        help='Directory written by fetch_afdb_structures.py.')
    parser.add_argument('--out_dir', required=True,
                        help='Where three_di.json and manifest_3di.json go.')
    parser.add_argument('--encoder', choices=['mini3di', 'foldseek'], default='mini3di',
                        help='mini3di (default) is pure Python and needs no binary. '
                             'foldseek is the reference implementation.')
    parser.add_argument('--foldseek_binary', default='foldseek')
    parser.add_argument('--limit', type=int, default=0,
                        help='Encode only the first N sequences, to smoke-test '
                             'the plumbing before committing to the full set.')
    args = parser.parse_args()

    generate(args.data_file, args.structures_dir, args.out_dir, args.encoder,
             args.foldseek_binary, args.limit)


if __name__ == '__main__':
    main()
