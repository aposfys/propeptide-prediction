'''
Download AlphaFold DB predicted structures for every UniProt accession in the
dataset, so ESM3 can be given a structure track alongside the sequence track.

Writes one .pdb per accession plus a manifest.json recording, for each
accession, whether a model exists, its global pLDDT, and whether the AFDB
sequence matches ours. The manifest is what make_embeddings_esm3_struct.py
reads; it never guesses from the filesystem.

Coverage is not total (~88% on a 25-accession sample). Proteins without a
usable structure are NOT dropped -- see make_embeddings_esm3_struct.py, which
falls back to a masked structure track so the dataset stays byte-identical to
the sequence-only runs. Dropping them would change the test partition and
break comparability with the ESM-2 baseline.

Note the file version: AFDB v4 URLs are retired and 404. The API reports
latestVersion (6 at time of writing), so we ask the API rather than hardcoding.
'''
import argparse
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd
from tqdm.auto import tqdm

API = 'https://alphafold.ebi.ac.uk/api/prediction/{acc}'


def _get(url: str, timeout: int, retries: int = 3, backoff: float = 2.0):
    '''GET with retries. Returns bytes, or None on a definitive 404.'''
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # genuinely absent, do not retry
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(backoff * (attempt + 1))
    return None


def fetch(data_file: str, out_dir: str, delay: float, timeout: int) -> dict:
    df = pd.read_csv(data_file)
    accessions = df['protein_id'].astype(str).tolist()
    sequences = dict(zip(df['protein_id'].astype(str), df['sequence'].astype(str)))
    print(f'{len(accessions)} accessions from {data_file}')

    manifest_path = os.path.join(out_dir, 'manifest.json')
    manifest = {}
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
        print(f'  resuming: {len(manifest)} already recorded')

    try:
        for acc in tqdm(accessions):
            if acc in manifest:
                continue

            pdb_path = os.path.join(out_dir, f'{acc}.pdb')
            entry = {'accession': acc, 'status': None, 'plddt': None,
                     'afdb_len': None, 'our_len': len(sequences[acc]),
                     'seq_match': None, 'version': None}

            try:
                raw = _get(API.format(acc=acc), timeout)
            except Exception as e:
                entry['status'] = f'api_error: {type(e).__name__}'
                manifest[acc] = entry
                continue

            if not raw:
                entry['status'] = 'no_model'
                manifest[acc] = entry
                continue

            meta = json.loads(raw)
            if not meta:
                entry['status'] = 'no_model'
                manifest[acc] = entry
                continue
            meta = meta[0]

            entry['plddt'] = meta.get('globalMetricValue')
            entry['version'] = meta.get('latestVersion')
            entry['afdb_len'] = len(meta.get('sequence', ''))
            # Exact match only. A near-match means a different isoform, and
            # per-residue structure tokens aligned to the wrong residues are
            # worse than no structure at all.
            entry['seq_match'] = meta.get('sequence') == sequences[acc]

            if not entry['seq_match']:
                entry['status'] = 'seq_mismatch'
                manifest[acc] = entry
                continue

            if not os.path.isfile(pdb_path):
                try:
                    body = _get(meta['pdbUrl'], timeout)
                except Exception as e:
                    entry['status'] = f'download_error: {type(e).__name__}'
                    manifest[acc] = entry
                    continue
                if not body:
                    entry['status'] = 'pdb_missing'
                    manifest[acc] = entry
                    continue
                with open(pdb_path, 'wb') as fh:
                    fh.write(body)

            entry['status'] = 'ok'
            manifest[acc] = entry
            time.sleep(delay)
    finally:
        # Always persist, so an interrupted run resumes instead of restarting.
        json.dump(manifest, open(manifest_path, 'w'), indent=1)
        print(f'\nmanifest -> {manifest_path}')

    return manifest


def report(manifest: dict) -> None:
    from collections import Counter
    counts = Counter(v['status'] for v in manifest.values())
    total = len(manifest)
    print(f'\n=== coverage over {total} accessions ===')
    for status, n in counts.most_common():
        print(f'  {status:24} {n:6}  ({100*n/total:.1f}%)')

    ok = [v for v in manifest.values() if v['status'] == 'ok']
    if ok:
        pl = sorted(v['plddt'] for v in ok if v['plddt'] is not None)
        if pl:
            mid = pl[len(pl)//2]
            print(f'\n  pLDDT over {len(pl)} usable models: '
                  f'min {pl[0]:.1f}  median {mid:.1f}  max {pl[-1]:.1f}')
            for lo, hi in ((0, 50), (50, 70), (70, 90), (90, 101)):
                n = sum(1 for p in pl if lo <= p < hi)
                print(f'    pLDDT {lo:3}-{hi:3}: {n:6}  ({100*n/len(pl):.1f}%)')
        print(f'\n  {len(ok)} proteins will get a real structure track; '
              f'{total - len(ok)} will be masked.')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_file', default='data/labeled_sequences.csv',
                   help='CSV with protein_id (UniProt accession) and sequence columns.')
    p.add_argument('--out_dir', required=True,
                   help='Directory for the .pdb files and manifest.json.')
    p.add_argument('--delay', type=float, default=0.1,
                   help='Seconds between accessions. Be polite to EBI.')
    p.add_argument('--timeout', type=int, default=30)
    p.add_argument('--report_only', action='store_true',
                   help='Print coverage from an existing manifest, download nothing.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.report_only:
        mp = os.path.join(args.out_dir, 'manifest.json')
        if not os.path.isfile(mp):
            raise SystemExit(f'No manifest at {mp}')
        report(json.load(open(mp)))
        return

    report(fetch(args.data_file, args.out_dir, args.delay, args.timeout))


if __name__ == '__main__':
    main()
