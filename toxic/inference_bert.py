import argparse
from pathlib import Path
from functools import partial

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ToxicDataset, collate_examples, SortSampler
from .common import ON_KAGGLE, DATA_DIR, MODEL_DIR, ToxicBot, tokenize

DEVICE = torch.device('cuda')
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


def get_token_ids(df, tokenizer, is_bert=False):
    tokens_tmp = []
    for _, row in df.iterrows():
        if is_bert:
            tokens = tokenize(row, tokenizer)
            tokens = ["[CLS]"] + tokens
            tokens_tmp.append(
                tokenizer.convert_tokens_to_ids(tokens))
        else:
            tokens = tokenize(row, tokenizer)
            # print(tokens)
            tokens = tokens
            tokens_tmp.append(
                tokenizer.convert_tokens_to_ids(tokens))
    token_ids = np.array(tokens_tmp)
    return token_ids


def main(args):
    model_path = Path(args.model_path)
    tokenizer = joblib.load(
        str(model_path / f"{args.tokenizer_name}.jbl"))
    df_test = pd.read_csv(str(DATA_DIR / "test.csv"))
    tokens = get_token_ids(
        df_test, tokenizer, args.model_name.startswith("bert"))
    test_ds = ToxicDataset(df_test, tokens, labeled=False)
    df_test.drop("comment_text", axis=1, inplace=True)
    test_sampler = SortSampler(
        test_ds, key=lambda x: len(test_ds.tokens[x]))
    df_test = df_test.iloc[list(iter(test_sampler))]
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
    test_loader = DataLoader(
        test_ds,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=test_sampler
    )
    model = torch.load(str(model_path / f"{args.model_name}.pth")).to(DEVICE)
    bot = ToxicBot(
        checkpoint_dir=Path("/tmp/"),
        log_dir=Path("/tmp/"),
        model=model, train_loader=None,
        val_loader=None, optimizer=None,
        echo=False,
        criterion=None,
        avg_window=100,
        callbacks=[],
        pbar=not ON_KAGGLE,
        use_tensorboard=False,
        device=DEVICE
    )
    test_pred = torch.sigmoid(bot.predict(
        test_loader, return_y=False)).numpy()[:, 0]
    df_test["prediction"] = test_pred
    df_test.to_csv(str(MODEL_DIR / "submission.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-path', type=str, default="data/cache")
    arg('--model-name', type=str, default="bert-base-uncased")
    arg('--tokenizer-name', type=str, default="bert-base-uncased")
    arg('--batch-size', type=int, default=128)
    arg('--maxlen', type=int, required=True)
    arg('--mode', type=str, default="head")
    args = parser.parse_args()
    main(args)
