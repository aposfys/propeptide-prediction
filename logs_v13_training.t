Traceback (most recent call last):
  File "/data/apostolos/envs/deeppeptide/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'coordinates'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/apostolos/envs/deeppeptide/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data/apostolos/envs/deeppeptide/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data/apostolos/DeepPeptide_esm3/src/train_loop_crf.py", line 263, in <module>
    train(parse_arguments())
  File "/data/apostolos/DeepPeptide_esm3/src/train_loop_crf.py", line 109, in train
    train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions)
  File "/data/apostolos/DeepPeptide_esm3/src/train_loop_crf.py", line 34, in get_dataloaders
    train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=train_partitions, label_type=args.label_type)
  File "/data/apostolos/DeepPeptide_esm3/src/utils/dataset.py", line 482, in __init__
    coordinate_strings = data['coordinates'].tolist()
  File "/data/apostolos/envs/deeppeptide/lib/python3.10/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
  File "/data/apostolos/envs/deeppeptide/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: 'coordinates'
