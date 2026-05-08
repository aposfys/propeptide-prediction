import numpy as np
import pandas as pd
from .crf_label_utils import parse_coordinate_string, segment_list_to_positional_label_sequence, peptide_list_to_binary_label_sequence

def build_prodomain_targets_from_row(row, seq: str, max_len: int = 50, min_len: int = 5) -> np.ndarray:
    coords = row.get("propeptide_coordinates", "")
    if coords is None or coords == "" or (isinstance(coords, float) and pd.isna(coords)):
        return np.zeros(len(seq), dtype=np.int64)

    segs = parse_coordinate_string(str(coords), merge_overlaps=True)
    if not segs:
        return np.zeros(len(seq), dtype=np.int64)

    y = segment_list_to_positional_label_sequence(
        segs,
        protein_length=len(seq),
        start_state=1,
        max_len=max_len,
        min_len=min_len,
    ).astype(np.int64)

    y[y < 0] = 0
    y[y > max_len] = max_len
    return y

def build_prodomain_binary_targets_from_row(row, seq: str) -> np.ndarray:
    """
    Binary prodomain labels:
      0 = outside
      1 = inside prodomain segment(s)
    Uses 1-based inclusive coordinates from 'propeptide_coordinates'.
    """
    coords = row.get("propeptide_coordinates", "")
    if coords is None or coords == "" or (isinstance(coords, float) and pd.isna(coords)):
        return np.zeros(len(seq), dtype=np.int64)

    segs = parse_coordinate_string(str(coords), merge_overlaps=True)
    if not segs:
        return np.zeros(len(seq), dtype=np.int64)

    y = peptide_list_to_binary_label_sequence(segs, protein_length=len(seq), label_value=1).astype(np.int64)
    return y
