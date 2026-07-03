'''
CRF train loop — propeptide-only, ESM-2 embeddings.
- LR warmup + cosine decay
- patience-based early stopping on propeptide F1
- no marginals during training
- no train metrics
'''
import json
import pickle
from typing import Dict, List, Tuple
import os
from torch.utils.data import DataLoader

from .models import LSTMCNNCRF, SimpleLSTMCNNCRF, SelfAttentionCRF
from .utils import add_dict_to_writer, PrecomputedCSVForOverlapCRFDataset
#from .utils.metrics_cleaned import compute_metrics, compute_metrics_with_propeptides
from .utils.manuscript_metrics import compute_all_metrics
from torch.optim import Adam
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
global_step = 0


def get_dataloaders(args: argparse.Namespace, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4]) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=train_partitions)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=valid_partitions)
        test_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=test_partitions)

    print(f'Loaded data. {len(train_set)} train sequences (p.{train_partitions}), {len(valid_set)} validation sequences (p.{valid_partitions}), {len(test_set)} test sequences (p.{test_partitions}).')


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=valid_set.collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=test_set.collate_fn, num_workers=0)

    return train_loader, valid_loader, test_loader


def get_model(args: argparse.Namespace):

    if args.model == 'lstmcnncrf':
        model = LSTMCNNCRF(
            input_size=args.embedding_dim,
            num_labels=2,
            dropout_input=args.dropout,
            num_states=51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_simple':
        model = SimpleLSTMCNNCRF(
            input_size=args.embedding_dim,
            num_labels=2,
            dropout_input=args.dropout,
            num_states=2,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'selfattentioncrf':
        model = SelfAttentionCRF(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_labels=2,
            dropout_input=args.dropout,
            num_states=51,
            n_heads=args.num_filters,
            attn_dropout=args.conv_dropout,
        )
    else:
        raise NotImplementedError(args.model)

    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def train(args, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], is_initiated: bool = False):
    global global_step
    global_step = 0
    train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions)


    model = get_model(args)
    model = model.to(device)
    model.feature_extractor.biLSTM.flatten_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.out_dir)

    previous_best = -100000000000

    for epoch in range(args.epochs):

        train_loss, train_probs, train_preds, train_peptides, train_labels = run_dataloader(train_loader, model, optimizer, writer, do_train=True)

        valid_loss, valid_probs, valid_preds, valid_peptides, valid_labels = run_dataloader(valid_loader, model, optimizer, writer, do_train=False)
        valid_metrics = compute_all_metrics(valid_probs, valid_preds, valid_labels, valid_loader.dataset.names, valid_loader.dataset.data, windows=[3])[0]
        add_dict_to_writer(valid_metrics, writer, global_step, prefix='Valid')
        writer.add_scalar('Valid/loss', valid_loss, global_step=global_step)

        stopping_metric = valid_metrics['f1 propeptides']
        print(f'Epoch {epoch} completed. Validation loss {valid_loss:.2f}  val_f1={stopping_metric:.4f}', flush=True)

        if stopping_metric > previous_best:
            previous_best = stopping_metric
            best_val_metrics = valid_metrics
            pickle.dump((valid_probs, valid_preds, valid_labels, valid_loader.dataset.names), open(os.path.join(args.out_dir, 'valid_outputs.pickle'), 'wb'))
            valid_metrics['epoch'] = epoch
            json.dump(valid_metrics, open(os.path.join(args.out_dir, 'valid_metrics.json'), 'w'), indent=2)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))
    
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pt')))
    test_loss, test_probs, test_preds, test_peptides, test_labels = run_dataloader(test_loader, model, optimizer, writer, do_train=False)
    #test_metrics = compute_crf_metrics(test_probs, test_preds, test_peptides, test_labels, organism=test_loader.dataset.data['organism'])
    #test_metrics = metrics_fn(test_peptides, test_preds, test_loader.dataset.data['organism'])
    test_metrics = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [3])[0]
    add_dict_to_writer(test_metrics, writer, global_step, prefix='Test')
    writer.add_scalar('Test/loss', test_loss, global_step=global_step)
    print('Test complete.')
    pickle.dump((test_probs, test_preds, test_labels, test_loader.dataset.names), open(os.path.join(args.out_dir, 'test_outputs.pickle'), 'wb'))
    json.dump(test_metrics, open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=2)

    return best_val_metrics, test_metrics

    

def run_dataloader(loader: torch.utils.data.DataLoader, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    writer: SummaryWriter,
                    do_train: bool = True,
                ) -> Tuple[float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    '''
    Run a dataloader through the model. Collect predicted probabilitities and
    true labels. Can be used both for training and prediction.
    '''
    global global_step

    true = [] # peptide coordinates
    labels = [] # labels made from coordinates
    probs = [] # per-position probabilities
    preds = [] # viterbi paths
    epoch_loss = []

    if do_train:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(loader):
        
        model.zero_grad()

        embeddings, mask, label, peptides= batch
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        label = label.long().to(device)

        if do_train:
            pos_probs, pos_preds, loss = model(embeddings, mask, label, skip_marginals=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            writer.add_scalar('Train/loss', loss.item(), global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                pos_probs, pos_preds, loss = model(embeddings, mask, label)

        true.extend(peptides)
        probs.append(pos_probs.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        preds.extend(pos_preds)
        epoch_loss.append(loss.item())


    epoch_loss = sum(epoch_loss)/len(epoch_loss)

    return epoch_loss, probs, preds, true, labels





def parse_arguments():
    '''Parse arguments, prepare output directory and dump run configuration.'''
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_dir', type=str, help='Embeddings dir produced by `extract.py`', default = '/data3/fegt_data/embeddings/')
    p.add_argument('--data_file', '-df', type=str, help='Sequences with Graph-Part headers', default = 'data/uniprot_12052022_cv_5_50/labeled_sequences.csv')
    p.add_argument('--partitioning_file', '-pf', type=str, help='Graph-Part output. Assume train-val-test split.', default = 'data/uniprot_12052022_cv_5_50/graphpart_assignments.csv')
    p.add_argument('--embedding', '-em', type=str, help='Sequence embedding strategy.', default='precomputed')
    p.add_argument('--embedding_dim', '-ed', type=int, help='Sequence embedding dimension.', default=1280)

    p.add_argument('--model', '-m', type=str, default='lstmcnncrf')

    p.add_argument('--out_dir', '-od', type=str, help='name that will be added to the runs folder output', default='train_run')
    p.add_argument('--epochs', type=int, default=30, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', '-bs', type=int, default=100, help='samples that will be processed in parallel')

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--conv_dropout', type=float, default=0.1)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--num_filters', type=int, default=32)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--patience', type=int, default=10, help='early stopping patience (epochs without propeptide F1 improvement)')

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'config.json'), 'w'), indent=3)

    return args


if __name__ == '__main__':
    train(parse_arguments())