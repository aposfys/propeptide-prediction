'''
Code to generate state space labels from a binary labeled peptide annotation sequence.

States are hardcoded.
'''
from typing import List, Tuple
import numpy as np


def parse_coordinate_string(coordinate_string: str, merge_overlaps: bool=True) -> List[Tuple[int,int]]:
    
    peptides = []
    coordinates = coordinate_string.split(',')
    
    if coordinate_string == '':
        return []
    # Cases to handle
    # --------------------111111111-------- 
    # ----------------11111111------------- N-terminal overlap
    # ---------------------111------------- inside of peptide
    # ----------------1111111111111111----- contains peptide
    # -----------------------------111111-- C-terminal overlap
    coordinates_parsed = []
    for coords in coordinates:
        s, e = coords.split('-')
        s, e = s.lstrip('('), e.rstrip(')')
        coordinates_parsed.append((int(s), int(e)))

    # start to end, long to short.
    sort_fn = lambda x: (x[0], -(x[1]-x[0]))
    coordinates_sorted = sorted(coordinates_parsed, key = sort_fn)

    coordinates_merged = []
    if merge_overlaps:
        previous_end = -1
        previous_start = -1
        for start, end in coordinates_sorted:
            if start>=previous_end:
                # the new start position comes after the previous end. 
                # Save the old one and open a new peptide.
                coordinates_merged.append([previous_start, previous_end])

                previous_start = start
                previous_end = end
            else:
                # the new start position is contained in the previous peptide.
                # continue the previous peptide.
                previous_end = max(previous_end, end) # either expand or keep prev if this one is smaller
        
        # handle the last peptide.
        coordinates_merged.append([previous_start, previous_end])

        return coordinates_merged[1:] # we add (-1,-1) to the list in the loop.

    else:
        return coordinates_sorted



def peptide_list_to_binary_label_sequence(peptides: List[Tuple[int,int]], protein_length: int, label_value: int = 1):
    '''Transform a list of peptides into a 0-1 label sequence.'''
    label = np.zeros(protein_length)

    for start, end in peptides:
        label[start-1:end] =label_value

    return label

def peptide_list_to_label_sequence(peptides: List[Tuple[int,int]], protein_length: int, start_state: int = 1, max_len: int = 60, min_len: int = 5) -> np.ndarray:
    '''Tranform a list of peptides into a multistate label sequence. Cannot handle overlapping peptides.'''
    label = np.zeros(protein_length)

    for start, end in peptides:
        peptide_length = end - start + 1 #upper bound is inclusive.

        # A peptide longer than max_len has no path through the state space. The
        # arithmetic below does not notice: it runs the C-terminal counter past
        # the start of the range and emits NEGATIVE state indices, which the old
        # code reported with a printed 'Bad label!' and then returned anyway.
        # Silent label corruption behind a print statement is the worst of both
        # worlds, so refuse instead.
        #
        # Unreachable on the distributed benchmark, which Teufel et al. filtered
        # to 5..50 -- every one of its 8,201 propeptides fits. It becomes
        # reachable the moment anyone rebuilds the dataset from unfiltered
        # UniProt, where 21% of annotations are longer than 50. See GRAMMAR.md.
        if peptide_length > max_len:
            raise ValueError(
                f'propeptide {start}-{end} is {peptide_length} residues, longer '
                f'than the grammar\'s max_len of {max_len}. It has no '
                f'representation in a {max_len + 1}-state CRF. Raise '
                f'--max_peptide_len (and --embedding_dim stays unchanged), or '
                f'filter the dataset. See GRAMMAR.md.')
        if peptide_length < min_len:
            raise ValueError(
                f'propeptide {start}-{end} is {peptide_length} residues, shorter '
                f'than the grammar\'s min_len of {min_len}. Lower '
                f'--min_peptide_len (this costs no extra states) or filter the '
                f'dataset. See GRAMMAR.md.')

        peptide_label = np.concatenate(
            [ 
            np.arange(start_state, start_state+min_len-2),#np.arange(1, 4), # from start to first position with skip connections
            # (end_state -1) - (peptide_length - min_len)
            np.arange((start_state+max_len-2 - (peptide_length - min_len)), start_state+max_len) #np.arange( 59-(peptide_length-5) ,61) 
            ]
        )
        # e.g. peptide of len 5 -> 1,2,3,59, 60
        # e.g. peptide of len 11-> 1,2,3,53,54,55,56,57,58,59,60
        label[start-1:end] = peptide_label


    # Defensive: the two length guards above should make this unreachable. Kept
    # as an assertion rather than the original print, so a grammar bug cannot
    # reach the CRF as a negative tag index.
    if (label < 0).any():
        raise ValueError(
            f'negative state index in the label sequence for peptides {peptides}. '
            f'This is a grammar bug, not a data problem -- max_len={max_len}, '
            f'min_len={min_len}.')

    return label



def peptide_list_to_multilabel_matrix(peptides: List[Tuple[int,int]], protein_length: int) -> np.ndarray:


    # TODO how to handle no-peptide in the multilabel case. i.e. with overlaps perfect solution of A would conflict with solving B
    label = np.zeros((protein_length, 61))


    for start, end in peptides:
        peptide_length = end - start + 1


        peptide_label = np.concatenate(
            [ 
            np.arange(1, 4), # from start to first position with skip connections
            np.arange( 59-(peptide_length-5) ,61) 
            ]
        )
        # e.g. peptide of len 5 -> 1,2,3,59, 60
        # e.g. peptide of len 11-> 1,2,3,53,54,55,56,57,58,59,60

        # set the positions in the matrix to true.
        label[np.arange(start-1,end), peptide_label] = 1

    return label
