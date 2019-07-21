import os
import argparse
from pathlib import Path
from functools import partial

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ToxicDataset, collate_examples, SortSampler
from .bert_model import get_model
from .common import ON_KAGGLE, ToxicBot
from .metric import (
    compute_bias_metrics_for_model, get_final_metric, calculate_overall_auc, power_mean,
    convert_dataframe_to_bool
)

DEVICE = torch.device('cuda')
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


def main(args):
    model_path = Path(args.model_path)
    tokenizer = joblib.load(
        str(model_path / f"{args.tokenizer_name}.jbl"))
    df_valid, tokens_valid = joblib.load(
        str(model_path / f"{args.valid_name}.jbl"))
    valid_ds = ToxicDataset(df_valid, tokens_valid, labeled=True)
    val_sampler = SortSampler(
        valid_ds, key=lambda x: len(valid_ds.tokens[x]))
    df_valid = df_valid.iloc[list(iter(val_sampler))]
    print(df_valid.target.describe())

    collate_fn = partial(
        collate_examples,
        truncate_len=args.maxlen,
        pad=(
            tokenizer.special_tokens["<pad>"]
            if args.model_name.startswith("gpt2") else 0),
        closing_id=(
            tokenizer.special_tokens["<cls>"]
            if args.model_name.startswith("gpt2") else tokenizer.vocab["[SEP]"]),
        mode=args.mode
    )
    valid_loader = DataLoader(
        valid_ds,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=val_sampler
    )
    model = torch.load(
        str(model_path / f"{args.model_name}.pth")).to(DEVICE)
    # model = get_model('bert-base-uncased').to(DEVICE)
    # model.load_state_dict(model_base.state_dict())
    bot = ToxicBot(
        checkpoint_dir=Path("/tmp/"),
        log_dir=Path("/tmp/"),
        model=model, train_loader=None,
        val_loader=None, optimizer=None,
        echo=False,
        criterion=nn.BCEWithLogitsLoss(),
        avg_window=100,
        callbacks=[],
        pbar=not ON_KAGGLE,
        use_tensorboard=False,
        device=DEVICE
    )
    valid_pred = torch.sigmoid(bot.predict(
        valid_loader, return_y=False)).numpy()[:, 0]
    df_valid["pred"] = valid_pred
    df_valid = convert_dataframe_to_bool(df_valid)
    df_bias_metrics = compute_bias_metrics_for_model(
        df_valid, IDENTITY_COLUMNS, 'pred', 'target')
    # more options can be specified also
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_bias_metrics)
    overall_auc = calculate_overall_auc(df_valid, "pred")
    print(f"Overall AUC: {overall_auc:.6f}")
    print(
        f"Mean bnsp_auc: {power_mean(df_bias_metrics['bnsp_auc'], -5):.6f}")
    print(
        f"Mean bpsn_auc: {power_mean(df_bias_metrics['bpsn_auc'], -5):.6f}")
    print(
        f"Mean subgroup auc: {power_mean(df_bias_metrics['subgroup_auc'], -5):.6f}")
    final = get_final_metric(df_bias_metrics, overall_auc)
    print(f"Final score: {final:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-path', type=str, default="data/cache")
    arg('--model-name', type=str, default="bert-base-uncased")
    arg('--tokenizer-name', type=str, default="bert-base-uncased")
    arg('--valid-name', type=str, default="valid")
    # arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--batch-size', type=int, default=128)
    arg('--maxlen', type=int, required=True)
    arg('--mode', type=str, default="head")
    args = parser.parse_args()
    main(args)
