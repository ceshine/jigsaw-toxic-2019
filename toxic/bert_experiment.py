from collections import deque
import os
import gc
import argparse
from pathlib import Path
from functools import partial
from itertools import chain

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from helperbot import (
    TriangularLR,
    LearningRateSchedulerCallback, GradualWarmupScheduler
)
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from .dataset import ToxicDataset, SortishSampler, SortSampler, AUX_COLUMNS
from .bert_model import get_model as get_bert_model
from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender
from .common import (
    ON_KAGGLE, DATA_DIR, MODEL_DIR, CACHE_DIR, ToxicBot, tokenize
)
from .loss import WeightedBCELossWithLogit, WeightedFocalLoss
from .metric import ToxicMetric, convert_dataframe_to_bool
from .preprocessing import IDENTITY_COLUMNS
from .metric import *


SEED = int(os.environ.get("SEED", "858"))
DEVICE = torch.device('cuda')

NO_DECAY = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

CACHE_DIR.mkdir(exist_ok=True)


def get_data(args):
    df_train = pd.read_csv(
        str(DATA_DIR / "train.csv")
    )
    print('loaded %d records' % len(df_train))
    # TODO: maybe remove no-good examples?
    df_train['comment_text'] = df_train['comment_text'].astype(str)
    df_train["comment_text"] = df_train["comment_text"].fillna(
        "DUMMY_VALUE")
    df_train = df_train.fillna(0)
    # convert target to 0,1
    df_train["weight"] = 1
    df_train.loc[:, AUX_COLUMNS] = (
        np.abs(2.0 * df_train[AUX_COLUMNS].values - 1.0) ** 0.5 *
        np.sign(df_train[AUX_COLUMNS].values - 0.5) + 1
    ) / 2
    return df_train


def get_tokens(df, tokenizer, max_length):
    tokens_tmp = []
    for _, row in df.iterrows():
        tokens = tokenize(row, tokenizer)
        if len(tokens) > max_length - 2:
            tokens = (
                ["[CLS]"] + tokens[:(max_length-2)//2] +
                tokens[-(max_length-2)//2:] + ["[SEP]"]
            )
        else:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        tokens_tmp.append(
            tokenizer.convert_tokens_to_ids(tokens)
        )
        assert tokens_tmp[-1][-1] == tokenizer.vocab["[SEP]"]
    token_ids = np.array(tokens_tmp)
    return token_ids


def get_loaders(args, tokenizer):
    df_train = get_data(args)
    if args.sample_size > 0:
        np.random.seed(42)
        sampled_idx = np.random.choice(
            np.arange(df_train.shape[0]), args.sample_size, replace=False
        )
        # print(sampled_idx[:5])
        np.random.seed(SEED)
        df_train = df_train.iloc[sampled_idx].reset_index(drop=True)
    print("Creating tokens...")
    tokens = get_tokens(df_train, tokenizer, args.maxlen)
    print("Done creating tokens!")
    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, valid_index) in enumerate(
            skf.split(df_train, (df_train["target"] >= 0.5).values.astype(np.uint8))):
        if i != args.fold:
            continue
        df_valid = df_train.iloc[valid_index].reset_index(drop=True)
        df_train = df_train.iloc[train_index].reset_index(drop=True)
        # print(args.fold, valid_index[:5], df_train["target"].values[:10])
        # print((df_valid[IDENTITY_COLUMNS] >= 0.5).sum(axis=0))
        tokens_valid = tokens[valid_index]
        tokens_train = tokens[train_index]
        break
    print(
        df_valid.shape[0],
        (df_valid.target == 0.5).sum(),
        (df_valid.target > 0.5).sum(),
        (df_valid.target <= 0.5).sum()
    )
    # print(df_valid[["target", "comment_text"]].head())
    train_ds = ToxicDataset(
        df_train, tokens_train, labeled=True)
    valid_ds = ToxicDataset(
        df_valid, tokens_valid, labeled=True)
    # Train dataset
    collat_fn = partial(
        collate_examples,
        truncate_len=args.maxlen,
        pad=0,
        mode=args.mode
    )
    trn_sampler = SortishSampler(
        train_ds, key=lambda x: len(df_train.comment_text[x].split()),
        bs=args.batch_size, chunk_size=200
    )
    train_loader = DataLoader(
        train_ds,
        collate_fn=collat_fn,
        batch_size=args.batch_size,
        sampler=trn_sampler,
        num_workers=0,
        pin_memory=True
    )
    # Valid dataset
    collat_fn = partial(
        collate_examples,
        truncate_len=320,
        pad=0,
        mode=args.mode
    )
    val_sampler = SortSampler(
        valid_ds, key=lambda x: len(df_valid.comment_text[x].split()))
    valid_loader = DataLoader(
        valid_ds,
        collate_fn=collat_fn,
        batch_size=args.batch_size * 4,
        num_workers=0,
        sampler=val_sampler,
        pin_memory=True
    )
    df_valid = df_valid.iloc[list(iter(val_sampler))]
    return train_loader, valid_loader, df_valid


class BertClassification(BertForSequenceClassification):
    def forward(self, input_ids, *args):
        return super().forward(input_ids, attention_mask=(input_ids > 0))


def collate_examples(batch, pad, truncate_len=250, mode="head"):
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
        tokens[i, :len(row)] = row
    # assert np.sum(tokens == closing_id) == len(batch)
    token_tensor = torch.from_numpy(tokens)
    # Labels
    if transposed[1][0] is None:
        return token_tensor, None
    weights = torch.FloatTensor(transposed[1]).unsqueeze(1)
    labels = torch.FloatTensor(transposed[2])
    return token_tensor, torch.cat([weights, labels], dim=1)


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Bert Finetuning")
def main(args):
    tokenizer = BertTokenizer.from_pretrained(
        args.model,
        do_lower_case=args.model.split("-")[-1] == "uncased",
    )
    # model = BertForSequenceClassification.from_pretrained(
    #     args.model, num_labels=len(AUX_COLUMNS)
    # ).to(DEVICE)

    model = BertClassification.from_pretrained(
        args.model, num_labels=len(AUX_COLUMNS)
    ).to(DEVICE)

    train_loader, valid_loader, df_valid = get_loaders(args, tokenizer)

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in NO_DECAY)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in NO_DECAY)], 'weight_decay': 0.0}
    ]

    n_steps = len(train_loader) * args.epochs
    optimizer = BertAdam(
        # model.parameters(),
        optimizer_grouped_parameters,
        lr=2e-5, warmup=0.05, weight_decay=0.01,
        t_total=n_steps
    )
    use_amp = False
    if args.amp and APEX_AVAILABLE:
        use_amp = True
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.amp
        )

    # model = model.train()
    # loss_fn = WeightedBCELossWithLogit()
    # accumulation_steps = args.grad_accu
    # train_losses = deque(maxlen=len(train_loader)//8)
    # avg_loss = 0.
    # avg_accuracy = 0.
    # lossf = None
    # optimizer.zero_grad()
    # tk0 = tqdm(train_loader, leave=False)
    # for i, (x_batch, y_batch) in enumerate(tk0):
    #     # w_batch, y_batch = y_batch[:, 0], y_batch[:, 1]
    #     y_pred = model(x_batch.to(DEVICE))
    #     loss = loss_fn(
    #         y_pred.to(DEVICE), y_batch.to(DEVICE)
    #     ) / accumulation_steps
    #     loss.backward()
    #     if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
    #         optimizer.step()                            # Now we can do an optimizer step
    #         optimizer.zero_grad()
    #     if lossf:
    #         lossf = 0.98*lossf+0.02*loss.item()*accumulation_steps
    #     else:
    #         lossf = loss.item()
    #     train_losses.append(loss.item()*accumulation_steps)
    #     if (i + 1) % (len(train_loader)//8) == 0:
    #         print(np.mean(train_losses))
    #     tk0.set_postfix(loss=np.mean(train_losses))
    #     avg_loss += loss.item()*accumulation_steps / len(train_loader)
    # print(lossf, avg_loss, np.mean(train_losses))

    bot = ToxicBot(
        checkpoint_dir=CACHE_DIR / "model_cache/",
        log_dir=Path(".") if ON_KAGGLE else CACHE_DIR / "logs/",
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        clip_grad=0,
        optimizer=optimizer,
        echo=not ON_KAGGLE,
        criterion=WeightedBCELossWithLogit(),
        avg_window=len(train_loader) // 8,
        callbacks=[],
        pbar=not ON_KAGGLE,
        use_tensorboard=False,
        device=DEVICE,
        use_amp=use_amp,
        gradient_accumulation_steps=args.grad_accu
    )
    bot.metrics.append(ToxicMetric(df_valid))
    # # bot.logger.info(bot.criterion)
    # bot.logger.info("Start fine-tuning...")
    bot.train(
        n_steps,
        log_interval=len(train_loader) // 8,
        snapshot_interval=len(train_loader),
        keep_n_snapshots=1
    )
    # # bot.load_model(bot.best_performers[0][1])
    # # bot.remove_checkpoints(keep=0)

    # Run validation
    # The following 2 lines are not needed but show how to download the model for prediction
    test_preds = []
    mx = 320
    model = model.eval()
    tk0 = tqdm(valid_loader, leave=True)
    tranct = 0
    with torch.no_grad():
        for i, (x_batch, _) in enumerate(tk0):
            y_pred = model(x_batch.to(DEVICE))
            test_preds.append(torch.sigmoid(y_pred[:, 0].cpu()))
            tranct = tranct + args.batch_size * 4 * (x_batch.shape[1] == mx)
            tk0.set_postfix(trunct=tranct, gpu_memory=torch.cuda.memory_allocated(
            ) // 1024 ** 2, batch_len=x_batch.shape[1])
    test_preds = torch.cat(test_preds, dim=0).numpy()
    MODEL_NAME = 'model1'
    df_valid[MODEL_NAME] = test_preds
    df_valid = convert_dataframe_to_bool(df_valid)
    bias_metrics_df = compute_bias_metrics_for_model(
        df_valid, IDENTITY_COLUMNS, MODEL_NAME, 'target')
    overall_auc = calculate_overall_auc(df_valid, MODEL_NAME)
    final = get_final_metric(bias_metrics_df, overall_auc)
    print(f"Overall AUC: {overall_auc:.6f}")
    print(f"Mean bnsp_auc: {power_mean(bias_metrics_df['bnsp_auc'], -5):.6f}")
    print(f"Mean bpsn_auc: {power_mean(bias_metrics_df['bpsn_auc'], -5):.6f}")
    print(
        f"Mean subgroup auc: {power_mean(bias_metrics_df['subgroup_auc'], -5):.6f}")
    print(f"Final score: {final:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default="bert-base-uncased")
    arg('--batch-size', type=int, default=32)
    arg('--sample-size', type=int, default=-1)
    arg('--start-layer', type=int, default=-4)
    arg('--grad-accu', type=int, default=1)
    arg('--alpha', type=float, default=.5)
    arg('--gamma', type=float, default=.25)
    arg('--maxlen', type=int, default=250)
    arg('--base-lr', type=float, default=1e-4)
    arg('--lr-decay', type=float, default=0.46416)
    arg('--weight-decay', type=float, default=0.1)
    arg('--weight-config', type=str, default="kaggle")
    # arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--epochs', type=float, default=1.)
    arg('--amp', type=str, default="")
    arg('--fold', type=int, default=0)
    arg('--mode', type=str, default="head")
    arg('--train-embd', action="store_true")
    args = parser.parse_args()
    main(args)
