'''
Functions to compute performance metrics.
Modified to align with 2nd-label task (Propeptide states 1-50).
'''

import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Tuple
from tqdm.auto import tqdm

# FIXED: Re-aligned to match the new 2-label model output
# Standard peptides are removed, so we set them to invalid states
PEPTIDE_START_STATE, PEPTIDE_END_STATE = -1, -1 
# Propeptides are now the primary branch, starting at state 1
PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE = 1, 50


def convert_path_to_peptide_borders(pred: List[int], start_state, stop_state, offset: int=0) -> List[Tuple[int,int]]:
    '''
    Given a sequence of states, find the borders of contiguous peptide segments.
    '''
    if start_state == -1: # Early exit if branch is disabled
        return []

    seq_peptides = []
    is_peptide = False
    peptide_start = 0

    for pos, p in enumerate(pred):
        if p == start_state and not is_peptide: # open a new peptide
            is_peptide = True
            peptide_start = pos
        elif p == stop_state and is_peptide: # close the peptide
            is_peptide = False
            seq_peptides.append((peptide_start + offset, pos + offset))
            
    if is_peptide: # handle sequence-ending peptides
        seq_peptides.append((peptide_start + offset, len(pred) - 1 + offset))
        
    return seq_peptides

def parse_coordinate_string(coordinate_string: str, merge_overlaps: bool=True) -> List[Tuple[int,int]]:
    if not coordinate_string or coordinate_string == '':
        return []
    
    coordinates = coordinate_string.split(',')
    coordinates_parsed = []
    for coords in coordinates:
        try:
            s, e = coords.split('-')
            s, e = s.replace('(', '').strip(), e.replace(')', '').strip()
            coordinates_parsed.append((int(s), int(e)))
        except ValueError:
            continue

    sort_fn = lambda x: (x[0], -(x[1]-x[0]))
    coordinates_sorted = sorted(coordinates_parsed, key = sort_fn)

    if merge_overlaps:
        if not coordinates_sorted: return []
        coordinates_merged = []
        curr_start, curr_end = coordinates_sorted[0]
        for next_start, next_end in coordinates_sorted[1:]:
            if next_start >= curr_end:
                coordinates_merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
            else:
                curr_end = max(curr_end, next_end)
        coordinates_merged.append((curr_start, curr_end))
        return coordinates_merged
    return coordinates_sorted

def get_counts_for_protein(true_coords: List[Tuple[int,int]], pred_coords: List[Tuple[int,int]], tolerance: int = 3) -> Tuple[int,int,int]:
    if len(pred_coords) == 0:
        return 0, len(true_coords), 0
    if len(true_coords) == 0:
        return 0, 0, len(pred_coords)

    # Convert to DF for overlap grouping
    true_df = pd.DataFrame(true_coords, columns=['start', 'stop'])
    true_df = true_df.sort_values(['start', 'stop'], ascending=[True, False])
    true_df['group'] = (true_df['stop'].cummax().shift() < true_df['start']).cumsum()
    true_df['matched'] = False

    pred_df = pd.DataFrame(pred_coords, columns=['start', 'stop'])
    pred_df['matched'] = False

    for t_idx, t_row in true_df.iterrows():
        for p_idx, p_row in pred_df.iterrows():
            start_match = abs(p_row['start'] - t_row['start']) <= tolerance
            stop_match = abs(p_row['stop'] - t_row['stop']) <= tolerance
            if start_match and stop_match:
                true_df.at[t_idx, 'matched'] = True
                pred_df.at[p_idx, 'matched'] = True
                break
    
    true_matched = true_df.groupby('group')['matched'].any()
    tp = true_matched.sum()
    fn = len(true_matched) - tp
    fp = (~pred_df['matched']).sum()

    return tp, fn, fp

def compute_peptide_finding_metrics(true_list, pred_list, tolerance: int = 3):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    for true, pred in zip(true_list, pred_list):
        tp, fn, fp = get_counts_for_protein(true, pred, tolerance)
        true_positives += tp
        false_negatives += fn
        false_positives += fp
    
    prec = (true_positives/(true_positives+false_positives)) if (true_positives+false_positives) >0 else 0
    recall = (true_positives/(true_positives+false_negatives)) if (true_positives+false_negatives)>0 else 0
    f1 = (2 * prec * recall) / (prec+recall) if (prec+recall) >0 else 0
    return prec, recall, f1

def compute_all_metrics(probs, preds, labels, names, true_df, windows: List[int] = [3]):
    # FIXED: Extracting propeptide borders from the new state 1-50 ladder
    propeptide_borders = [convert_path_to_peptide_borders(pred, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE, offset=1) for pred in preds]

    prediction_df = pd.DataFrame({'pred_propeptides': propeptide_borders}, index=names)
    df = prediction_df.join(true_df[['true_propeptides']], how='inner')
    
    metrics = []
    for tolerance in windows:
        prec_pro, rec_pro, f1_pro = compute_peptide_finding_metrics(
            df['true_propeptides'].tolist(), 
            df['pred_propeptides'].tolist(), 
            tolerance=tolerance
        )

        metrics.append({
            'f1 peptides': 0.0, # Placeholder for training loop compatibility
            'precision propeptides': prec_pro,
            'recall propeptides': rec_pro,
            'f1 propeptides': f1_pro,
        })
    
    return metrics

# def main():

#     # for each model, I want (prec, recall, f1) * (0,1,2,3) * (pep, propep, merged)

#     df = pd.read_csv('../data/uniprot_12052022_cv_5_50/labeled_sequences.csv', index_col='protein_id')
#     df = df.fillna('') # empty coordinates would become nan.
#     coordinate_strings = df['coordinates'].tolist()
#     propeptide_coordinate_strings = df['propeptide_coordinates'].tolist()
#     coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in coordinate_strings]
#     propeptide_coordinates = [parse_coordinate_string(x, merge_overlaps=False) for x in propeptide_coordinate_strings]
#     df['true_peptides'] = coordinates
#     df['true_propeptides'] = propeptide_coordinates

#     metrics_dfs = []
#     for checkpoint in tqdm(BEST_CHECKPOINTS):

#         metrics = score_one_model(os.path.join(checkpoint, 'test_outputs.pickle'), df)
#         metrics_df = pd.DataFrame.from_dict(metrics)
#         metrics_df.index = pd.MultiIndex.from_product([metrics_df.index, [checkpoint]], names=['tolerance', 'model'])
#         metrics_dfs.append(metrics_df)

    
#     metrics_df = pd.concat(metrics_dfs).sort_index()

#     means = metrics_df.groupby(level=0).mean()
#     means.to_csv('crf_model_means.csv')
#     metrics_df.to_csv('crf_model_all_cv.csv')


# if __name__ == '__main__':
#     main()