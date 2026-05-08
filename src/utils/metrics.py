"""
Compatibility shim for DeepPeptide training loop.

Provides:
- add_dict_to_writer(d, writer, step, prefix="")
- compute_crf_metrics(...)
- compute_metrics(...)
"""

from __future__ import annotations
from typing import Dict, Any

def add_dict_to_writer(d: Dict[str, Any], writer, step: int, prefix: str = "") -> None:
    """
    Log a metrics dict to TensorBoard SummaryWriter.
    Safe no-op if writer is None or d is not a dict.
    """
    if writer is None or not isinstance(d, dict):
        return

    for k, v in d.items():
        try:
            tag = f"{prefix}/{k}" if prefix else str(k)
            writer.add_scalar(tag, float(v), step)
        except Exception:
            pass

def compute_crf_metrics(*args, **kwargs) -> Dict[str, float]:
    return {}

def compute_metrics(*args, **kwargs) -> Dict[str, float]:
    return {}
