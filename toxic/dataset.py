import gc

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

AUX_COLUMNS = [
    'target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat',
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


class ToxicDataset(Dataset):
    """Custom GAP Dataset class"""

    def __init__(self, df, tokens, labeled=True):
        self.labeled = labeled
        if labeled:
            self.y = df[AUX_COLUMNS].fillna(0).values
            self.weight = df.weight.values
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.weight[idx], self.y[idx]
        return self.tokens[idx], None, None


def collate_examples(batch, pad, closing_id, truncate_len=250, mode="head"):
    """Batch preparation.

    1. Pad the sequences
    """
    transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])) + 1,
        truncate_len
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64) + pad
    # print(pad, transposed[0][-1])
    for i, row in enumerate(transposed[0]):
        if mode == "head":
            row = row[:max_len-1]
        elif mode == "both":
            if len(row) <= max_len:
                row = row[:max_len-1]
            else:
                row = (
                    row[:((truncate_len) // 2 - 1)] +
                    row[-((truncate_len) // 2):]
                )
        tokens[i, :len(row)] = row
        tokens[i, len(row)] = closing_id
    assert np.sum(tokens == closing_id) == len(batch)
    token_tensor = torch.from_numpy(tokens)
    # Labels
    if transposed[1][0] is None:
        return token_tensor, None
    weights = torch.FloatTensor(transposed[1]).unsqueeze(1)
    labels = torch.FloatTensor(transposed[2])
    return token_tensor, torch.cat([weights, labels], dim=1)


class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.

    Taken from fastai library.
    """

    def __init__(self, data_source, key, bs, chunk_size=50):
        self.data_source, self.key, self.bs = data_source, key, bs
        self.chunk_size = 50

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        while True:
            idxs = np.random.permutation(len(self.data_source))
            sz = self.bs * self.chunk_size
            ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
            sort_idx = np.concatenate(
                [sorted(s, key=self.key, reverse=True) for s in ck_idx])
            sz = self.bs
            ck_idx = [sort_idx[i:i+sz]for i in range(0, len(sort_idx), sz)]
            # find the chunk with the largest key,
            max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])
            # then make sure it goes first.
            if len(ck_idx[max_ck]) != self.bs:
                continue
            ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
            sort_idx = np.concatenate(np.random.permutation([
                np.random.permutation(chunk.reshape(self.bs, -1)).reshape(-1)
                for chunk in ck_idx[1:-1]
            ]))
            sort_idx = np.concatenate((ck_idx[0], sort_idx, ck_idx[-1]))
            break
        return iter(sort_idx)


class SortSampler(Sampler):
    """
    Taken from fastai library.
    """

    def __init__(self, data_source, key):
        self.data_source, self.key = data_source, key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))
