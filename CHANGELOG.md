# Changelog: DeepPeptide Refactoring to Propeptide Predictor

This document details the modifications made to the open-source DeepPeptide model in order to refactor it to act as a dedicated propeptide predictor (binary region classification: Propeptide vs. None).

## 1. State Space Mapping & CRF Reduction

**Files Modified**: 
- `src/models/crf_models.py`
- `predictor/model.py`

**Modifications**:
- **num_states**: Changed the default `num_states` from `61` (or `101`) to `51` across `CRFBaseModel`, `LSTMCNNCRF`, and `SelfAttentionCRF`.
  - *Old*: `num_states = 61`
  - *New*: `num_states = 51`
- **num_labels**: Changed the default from `3` (None, Peptide, Propeptide) to `2` (None, Propeptide).
- **_repeat_emissions**: Removed the mapping for a third emission label. Now strictly maps `emissions[:,:,1]` (Propeptide) to states `1-50`.
  - *Old*:
    ```python
    elif emissions.shape[-1] == 3:
        emissions_out = torch.zeros(...)
        emissions_out[:,:,0] = emissions[:,:,0]
        emissions_out[:,:, 1:] = emissions[:,:,1].unsqueeze(-1)
        emissions_out[:,:, (self.max_len+1):] = emissions[:,:,2].unsqueeze(-1)
    ```
  - *New*: Block removed completely. Only `emissions.shape[-1] == 2` block retained.
- **get_crf_constraints**: Removed the parameter `n_branches` and completely dropped the multi-branch logic that separated states `1-50` (Peptides) and `51-100` (Propeptides). Only one branch remains, assigning states `1-50` to Propeptide.
  - *Old*: `if n_branches == 2: ...`
  - *New*: Block removed completely.

---

## 2. Data Pipeline Updates

**File Modified**: 
- `src/utils/dataset.py`

**Modifications**:
- **PrecomputedCSVForOverlapCRFDataset.__getitem__**: Completely eradicated mature peptide annotations by passing an empty list `[]` to `_sample_from_overlapping_peptides` instead of `peptides`. This forces the model to ignore regular mature peptides. Mapped propeptide annotations to states `1-50`.
  - *Old*: 
    ```python
    peptides, propeptides = self._sample_from_overlapping_peptides(peptides, propeptides)
    label = peptide_list_to_label_sequence(peptides, seq_len, max_len = 50) 
    propeptide_label = peptide_list_to_label_sequence(propeptides, seq_len, start_state=51, max_len=50) 
    label = label + propeptide_label
    ```
  - *New*:
    ```python
    # Ignore regular peptides by passing an empty list for peptides
    _, propeptides = self._sample_from_overlapping_peptides([], propeptides)

    # Propeptides mapped to states 1-50
    propeptide_label = peptide_list_to_label_sequence(propeptides, seq_len, start_state=1, max_len=50) 
    label = torch.from_numpy(propeptide_label)
    ```

---

## 3. Metrics Updates

**File Modified**: 
- `src/utils/manuscript_metrics.py`

**Modifications**:
- Updated `PROPEPTIDE_START_STATE` and `PROPEPTIDE_END_STATE` to `1` and `50` respectively.
- Completely removed `PEPTIDE_START_STATE`, `PEPTIDE_END_STATE`, and associated evaluation logic. The validation F1 calculations now focus solely on Propeptide.
  - *Old*: 
    ```python
    PEPTIDE_START_STATE, PEPTIDE_END_STATE = 1, 50
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE = 51, 100
    ...
    prec_pep, rec_pep, f1_pep = compute_peptide_finding_metrics(...)
    prec_all, rec_all, f1_all = compute_peptide_finding_metrics(...)
    ```
  - *New*:
    ```python
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE = 1, 50
    ...
    prec_pro, rec_pro, f1_pro = compute_peptide_finding_metrics(true, pred, tolerance=tolerance)
    ```

---

## 4. Training Loop Simplification

**File Modified**: 
- `src/train_loop_crf.py`

**Modifications**:
- **get_model**: Explicitly instantiated all models with `num_labels=2` and `num_states=51`. Removed dynamic logic parsing `label_type`.
- **Early Stopping**: Switched the `stopping_metric` from monitoring the average of peptide and propeptide F1 scores to solely monitoring propeptide F1 scores.
  - *Old*: `stopping_metric = (valid_metrics['f1 peptides'] + valid_metrics['f1 propeptides'])/2`
  - *New*: `stopping_metric = valid_metrics['f1 propeptides']`

---

## 5. CPU Optimizations & Redundancy

**File Modified**: 
- `predictor/predict.py`

**Modifications**:
- Disabled `compute_marginal_probabilities` by commenting out its calls. This eliminates the forward-backward algorithm computation entirely to preserve CPU resources during inference.
- Commented out data handling, processing, and output mechanisms related strictly to marginal probability and sequence plots (e.g. PNG outputs, probabilities storage). Retained JSON logic, focusing strictly on Propeptide output and passing data to `write_fancy_output()`.
  - *Old*: 
    ```python
    if compute_marginals:
        marginals = model.crf.compute_marginal_probabilities(emissions, mask)
        batch_marginals.append(marginals.cpu())
    ...
    if args.output_fmt == 'img':
        # logic generating images and probability matrices
    ```
  - *New*: Codeblocks above are completely commented out to optimize inference strictly for the core output.