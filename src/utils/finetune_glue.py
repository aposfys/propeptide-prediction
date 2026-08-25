'''
Glue between PLMBackbone and the existing DeepPeptide CRF head.

Three pieces:
  SequenceCRFDataset  -- yields raw sequences instead of cached .pt embeddings
  FineTunedCRF        -- backbone + input LayerNorm + the unchanged LSTMCNNCRF
  LengthBucketSampler -- batches similar-length sequences together

Lives in src/utils/ because it shares crf_label_utils with dataset.py; import
PLMBackbone from src.models.plm_backbone at the call site.

CROPPING IS NOT FREE, AND THAT IS DELIBERATE
--------------------------------------------
__getitem__ crops sequences to max_len and drops propeptide segments that fall
past the boundary, so the label sequence always matches the cropped sequence.
`self.data['true_propeptides']` deliberately keeps the UNCROPPED annotation,
because that is what compute_all_metrics scores against. A propeptide living in
a discarded tail therefore becomes an unavoidable false negative rather than
disappearing from the denominator.

That is the honest accounting -- the model is judged on the whole protein -- but
it means max_len is a measurable handicap, not a free simplification. 137 of
8,449 sequences (1.6%) exceed 1,022 aa. The cached-embedding runs in RESULTS.md
had no such limit, so a fine-tuned number is not perfectly comparable to them
unless the frozen control is run through this same path (n_lora_blocks=0), which
is exactly why that control exists.
'''
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

from .crf_label_utils import parse_coordinate_string, peptide_list_to_label_sequence


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class SequenceCRFDataset(Dataset):
    '''Same partitions, labels and metadata as PrecomputedCSVForOverlapCRFDataset,
    but returns the sequence string so the encoder can run in the training graph.

    `.names` and `.data` are kept identical so `compute_all_metrics` and
    `measure_performance.py` work unchanged.
    '''

    def __init__(self, data_file, partitioning_file, partitions=(0,), max_len: int = 1024):
        super().__init__()
        data = pd.read_csv(data_file, index_col='protein_id')
        partitioning = pd.read_csv(partitioning_file, index_col='AC')
        data = data.join(partitioning)
        data = data.loc[data['cluster'].isin(list(partitions))]
        data = data.fillna('')

        self.max_len = max_len
        self.data = data
        self.names = data.index.tolist()
        self.sequences = data['sequence'].tolist()

        propeptides = [parse_coordinate_string(x, merge_overlaps=True)
                       for x in data['propeptide_coordinates'].tolist()]
        # Uncropped on purpose -- see the module docstring. This is the ground
        # truth the metric scores against.
        self.data['true_propeptides'] = propeptides
        self.data['true_peptides'] = [[] for _ in range(len(data))]
        self.propeptides = propeptides

        self.lengths = [min(len(s), max_len) for s in self.sequences]
        self.n_cropped = sum(1 for s in self.sequences if len(s) > max_len)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        seq = self.sequences[index][: self.max_len]
        propeptides = self.propeptides[index]
        # Cropping can truncate a propeptide that starts past max_len; keep only
        # segments fully inside the crop so labels and sequence stay consistent.
        if len(self.sequences[index]) > self.max_len:
            propeptides = [(s, e) for (s, e) in propeptides if e <= self.max_len]
        label = peptide_list_to_label_sequence(propeptides, len(seq), start_state=1, max_len=50)
        return seq, torch.from_numpy(label), propeptides

    @staticmethod
    def collate_fn(batch):
        seqs, labels, propeptides = zip(*batch)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return list(seqs), labels, list(propeptides)


class LengthBucketSampler(Sampler):
    '''Shuffle, then batch length-sorted within a large chunk.

    With a median length of 142 and a p99 of 1,156, random batching pads most
    batches to the longest member and wastes the majority of the compute. This
    keeps shuffling (a fixed length-sorted order would correlate batches with
    protein family) while cutting padding from ~3.0x to ~1.06x at batch 16.
    '''

    def __init__(self, lengths: Sequence[int], batch_size: int,
                 chunk_multiplier: int = 32, shuffle: bool = True, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.chunk = batch_size * chunk_multiplier
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        '''Explicit epoch control, for when the loop wants reproducible order.'''
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for start in range(0, len(idx), self.chunk):
            chunk = idx[start:start + self.chunk]
            chunk = chunk[np.argsort(self.lengths[chunk], kind='stable')]
            batches.extend(chunk[i:i + self.batch_size].tolist()
                           for i in range(0, len(chunk), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return int(np.ceil(len(self.lengths) / self.batch_size))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def align_labels(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''Trim/pad CRF labels to the encoder's output length.

    Label tensors are padded by collate_fn to the longest label in the batch;
    the encoder's output is padded to the longest tokenised sequence. Those
    agree whenever both crop at the same max_len, but a mismatch would shift
    every label in the batch by the offset and show up only as a mysteriously
    bad F1 -- so it is reconciled explicitly rather than left to broadcast.
    '''
    L = mask.shape[1]
    if labels.shape[1] > L:
        return labels[:, :L]
    if labels.shape[1] < L:
        pad = torch.zeros(labels.shape[0], L - labels.shape[1],
                          dtype=labels.dtype, device=labels.device)
        return torch.cat([labels, pad], dim=1)
    return labels


class FineTunedCRF(nn.Module):
    '''PLMBackbone -> LayerNorm -> existing LSTMCNNCRF.

    The LayerNorm is the important addition. `LSTMCNN` applies conv1 straight to
    the incoming features with no input normalisation, which is exactly why raw
    pre-norm ESM3 embeddings (per-token norm ~9,793 against ESM-2's ~10) drove
    90.7% biLSTM gate saturation. Normalising at the head boundary makes the
    head scale-agnostic: the same head then accepts ESM-2 or ESM3 features, and
    a future embedder swap cannot reintroduce that failure. It also removes a
    confound from any embedder comparison -- differences become representational
    rather than a matter of feature scale.
    '''

    def __init__(self, backbone, head, normalize_input: bool = True):
        super().__init__()
        self.backbone = backbone
        self.input_norm = nn.LayerNorm(backbone.d_model) if normalize_input else nn.Identity()
        self.head = head

    def forward(self, sequences: List[str], labels=None, skip_marginals: bool = True,
                use_focal: bool = False):
        reps, mask = self.backbone(sequences)          # (B, L, D), (B, L)
        reps = self.input_norm(reps)
        embeddings = reps.permute(0, 2, 1)             # head expects (B, D, L)
        if labels is not None:
            labels = align_labels(labels, mask)
        # The CRF's forward algorithm and Viterbi are log-space dynamic
        # programming over 51 states; bf16 loses too much mantissa there. Cast
        # emissions back to fp32 by keeping the head outside any autocast region
        # at the call site, or wrap only the backbone.
        return self.head(embeddings, mask, labels,
                         skip_marginals=skip_marginals, use_focal=use_focal)
