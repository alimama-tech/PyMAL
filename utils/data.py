import os
import json
import torch
import numpy as np


SEQ_LEN = 20
EPSILON = 1e-8


TRAIN_DATA = [
    'train-00.parquet', 'train-01.parquet', 'train-02.parquet', 'train-03.parquet', 'train-04.parquet',
    'train-05.parquet', 'train-06.parquet', 'train-07.parquet', 'train-08.parquet', 'train-09.parquet',
    'train-10.parquet', 'train-11.parquet', 'train-12.parquet', 'train-13.parquet', 'train-14.parquet',
    'train-15.parquet', 'train-16.parquet', 'train-17.parquet', 'train-18.parquet', 'train-19.parquet',
]
TEST_DATA = ['test.parquet']


USER_COLS = ['user_id', 'user_feat0', 'user_feat1', 'user_feat2', 'user_feat3', 'user_feat4', 'user_feat5']
AD_COLS = ['ad_feat0', 'ad_feat1', 'ad_feat2', 'ad_feat3', 'ad_feat4', 'ad_feat5', 'context_feat0', 'context_feat1', 'context_feat2', 'context_feat3']
MM_COLS = ['mm_feat0_seq', 'mm_feat1_seq']
SEQ_COLS = ['ad_feat0_seq', 'ad_feat1_seq', 'ad_feat2_seq']
LABEL_COLS = ['first', 'last', 'mta', 'linear']
MAP_COLS = {'ad_feat0_seq' : 'ad_feat0', 'ad_feat1_seq' : 'ad_feat1', 'ad_feat2_seq' : 'ad_feat2'}


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--seed', type=int, default=2026, help='Random Seed')
    parser.add_argument('--model', type=str, default='MoAE', help='Model Name')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch Size')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden Dimension')
    parser.add_argument('--lamda', type=float, default=0.1, help='Coefficient of Auxiliary Loss')
    parser.add_argument('--path', type=str, default='/home/zl428934/MAL_MM_PT/', help='Folder Path')
    parser.add_argument('--pcgrad', action='store_true', help='Enable PCGrad')
    parser.add_argument('--gcsgrad', action='store_true', help='Enable GCSGrad')
    parser.add_argument('--main_view', type=str, default='last', help='Main View')
    parser.add_argument('--aux_views', type=str, nargs='*', default=[], help='Auxiliary Views')
    args, _ = parser.parse_known_args()
    return args


def get_vocabs(path_vocabs):
    vocabs = {col: {} for col in USER_COLS + AD_COLS}
    for col in USER_COLS + AD_COLS:
        with open(os.path.join(path_vocabs, f"{col}.json"), 'r') as f:
            vocabs[col] = json.load(f)
    return vocabs


def df_to_dict(df, vocabs):
    result = {}
    for col in AD_COLS + USER_COLS:
        vocab = vocabs[col]
        mapper = np.vectorize(lambda x: vocab.get(x))
        result[col] = mapper(df[col].values).astype(np.int32)
    for col in SEQ_COLS:
        vocab = vocabs[MAP_COLS[col]]
        seq_mapped = []
        for seq in df[col].values:
            mapped = [vocab.get(token) for token in seq]
            seq_mapped.append(mapped)
        result[col] = np.array(seq_mapped, dtype=np.int32)
    for col in MM_COLS:
        values = df[col].values
        result[col] = np.stack([np.asarray(v, dtype=np.float32) for v in values], axis=0)
    for col in LABEL_COLS:
        result[col] = (df[col].values > 0).astype(np.int32)
    return result


class MALDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.len = len(next(iter(data.values())))
        self.data = data

        self.static_ad = np.stack([self.data[col] for col in AD_COLS], axis=1).astype(np.int32)
        self.static_user = np.stack([self.data[col] for col in USER_COLS], axis=1).astype(np.int32)
        mm_stacked = np.stack([self.data[col] for col in MM_COLS], axis=2)
        self.mm_arr = np.transpose(mm_stacked, (0, 2, 1)).astype(np.float32)
        seq_stacked = np.stack([self.data[col] for col in SEQ_COLS], axis=2)
        self.seq_arr = np.transpose(seq_stacked, (0, 2, 1)).astype(np.int32)

        self.static_ad = torch.from_numpy(self.static_ad).long()
        self.static_user = torch.from_numpy(self.static_user).long()
        self.mm_arr = torch.from_numpy(self.mm_arr).float()
        self.seq_arr = torch.from_numpy(self.seq_arr).long()
        self.labels = {col: torch.from_numpy(self.data[col]).long() for col in LABEL_COLS}

    def __getitem__(self, idx):
        return {
            'ad': self.static_ad[idx],
            'user': self.static_user[idx],
            'mm': self.mm_arr[idx],
            'seq': self.seq_arr[idx],
            'label': {k: v[idx] for k, v in self.labels.items()}
        }

    def __len__(self):
        return self.len
