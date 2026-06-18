from .dataset import PrecomputedCSVDataset, BLOSUMCSVDataset, PrecomputedCSVForCRFDataset, PrecomputedCSVForOverlapCRFDataset


def add_dict_to_writer(d, writer, step, prefix=''):
    for k, v in d.items():
        try:
            writer.add_scalar(f'{prefix}/{k}', float(v), global_step=step)
        except (TypeError, ValueError):
            pass