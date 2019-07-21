import os
import gc
import argparse
from pathlib import Path
from functools import partial
from itertools import chain

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer, BertAdam
from sklearn.model_selection import StratifiedKFold
from helperbot import (
    TriangularLR, WeightDecayOptimizerWrapper, freeze_layers,
    LearningRateSchedulerCallback, GradualWarmupScheduler
)
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from .dataset import ToxicDataset, collate_examples, SortishSampler, SortSampler, AUX_COLUMNS
from .bert_model import get_model as get_bert_model
from .gpt2_model import get_model as get_gpt2_model, SPECIAL_TOKENS
from .preprocessing import kaggle_data_weights, IDENTITY_COLUMNS, custom_data_weights, yuval_weights
from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender
from .common import (
    ON_KAGGLE, DATA_DIR, MODEL_DIR, CACHE_DIR, ToxicBot, tokenize,
    TargetFocalLoss, AuxFocalLoss
)
from .loss import WeightedBCELossWithLogit, WeightedFocalLoss
from .metric import ToxicMetric

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
    df_train['target_raw'] = df_train['target'].copy()
    df_train['target'] = (df_train['target'].fillna(0) >= 0.5).astype(np.uint8)
    if args.weight_config == "kaggle":
        df_train = kaggle_data_weights(df_train)
        df_train['target'] = df_train['target_raw']
    elif args.weight_config == "custom":
        df_train = custom_data_weights(df_train)
    elif args.weight_config == "yuval":
        df_train = yuval_weights(df_train)
        df_train['target'] = df_train['target_raw']
        df_train.loc[:, AUX_COLUMNS] = (
            np.abs(2.0 * df_train[AUX_COLUMNS].values - 1.0) ** 0.5 *
            np.sign(df_train[AUX_COLUMNS].values - 0.5) + 1
        ) / 2
    else:
        raise ValueError("Unrecognized weight config!")
    print(df_train.target.describe())
    # df_train["weight"] = 1
    # df_train['target'] = df_train['target_raw']
    # df_train.loc[:, AUX_COLUMNS] = (
    #     np.abs(2.0 * df_train[AUX_COLUMNS].values - 1.0) ** 0.5 *
    #     np.sign(df_train[AUX_COLUMNS].values - 0.5) + 1
    # ) / 2
    del df_train['target_raw']
    return df_train


def get_cached_tokens(args, df, tokenizer):
    cache_path = Path(CACHE_DIR / f"tokens_{args.model}.jbl")
    if cache_path.exists():
        token_ids = joblib.load(str(cache_path))
    else:
        tokens_tmp = []
        for _, row in df.iterrows():
            if args.model.startswith("bert"):
                tokens = tokenize(row, tokenizer)
                tokens = ["[CLS]"] + tokens
                tokens_tmp.append(
                    tokenizer.convert_tokens_to_ids(tokens)
                )
            else:
                tokens = tokenize(row, tokenizer)
                # print(tokens)
                tokens = tokens
                tokens_tmp.append(
                    tokenizer.convert_tokens_to_ids(tokens)
                )
                # print(tokens_tmp[-1])
                # print(tokenizer.convert_ids_to_tokens(tokens_tmp[-1]))
                # tokens_tmp.append(
                #     tokenizer.encode(
                #         row.comment_text.strip() + " <cls>"
                #     ))
        token_ids = np.array(tokens_tmp)
        joblib.dump(token_ids, str(cache_path))
    return token_ids


def get_loaders(args, tokenizer):
    df_train = get_data(args)
    tokens = get_cached_tokens(args, df_train, tokenizer)
    if args.sample_size > 0:
        np.random.seed(42)
        sampled_idx = np.random.choice(
            np.arange(df_train.shape[0]), args.sample_size, replace=False
        )
        np.random.seed(SEED)
        df_train = df_train.iloc[sampled_idx].reset_index(drop=True)
        tokens = tokens[sampled_idx]
    skf = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=42)
    for i, (train_index, valid_index) in enumerate(
            skf.split(df_train, (df_train["target"] >= 0.5).values.astype(np.uint8))):
        if i != args.fold:
            continue
        df_valid = df_train.iloc[valid_index].reset_index(drop=True)
        df_train = df_train.iloc[train_index].reset_index(drop=True)
        tokens_valid = tokens[valid_index]
        tokens_train = tokens[train_index]
        break
    train_ds = ToxicDataset(
        df_train, tokens_train, labeled=True)
    valid_ds = ToxicDataset(
        df_valid, tokens_valid, labeled=True)
    df_valid = df_valid[
        list(set(AUX_COLUMNS + ["weight"] + IDENTITY_COLUMNS))
    ]
    # Dump df_valid for later evaluation
    joblib.dump(
        [
            df_valid, tokens_valid
        ],
        str(MODEL_DIR /
            f"valid_{args.model}_{args.sample_size}_{args.weight_config}_f{args.fold}.jbl")
    )
    # Train dataset
    collat_fn = partial(
        collate_examples,
        truncate_len=args.maxlen,
        pad=tokenizer.special_tokens["<pad>"] if args.model == "gpt2" else 0,
        closing_id=tokenizer.special_tokens["<cls>"] if args.model == "gpt2" else tokenizer.vocab["[SEP]"],
        mode=args.mode
    )
    trn_sampler = SortishSampler(
        train_ds, key=lambda x: len(train_ds.tokens[x]),
        bs=args.batch_size, chunk_size=100
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
        truncate_len=args.maxlen if args.model == "gpt2" else 320,
        pad=tokenizer.special_tokens["<pad>"] if args.model == "gpt2" else 0,
        closing_id=tokenizer.special_tokens["<cls>"] if args.model == "gpt2" else tokenizer.vocab["[SEP]"],
        mode=args.mode
    )
    val_sampler = SortSampler(
        valid_ds, key=lambda x: len(valid_ds.tokens[x]))
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


def prepare_model(layers, learning_rates, weight_decays, start_layer=0):
    optimizer_grouped_parameters = list(chain.from_iterable(
        [
            {
                'params': [p for n, p in layers[i].named_parameters()
                           if not any(nd in n for nd in NO_DECAY)],
                'lr': learning_rates[i]
            },
            {
                'params': [p for n, p in layers[i].named_parameters()
                           if any(nd in n for nd in NO_DECAY)],
                'lr': learning_rates[i]
            }
        ] for i in range(start_layer, len(layers))
    ))
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(
            optimizer_grouped_parameters
        ),
        weight_decay=list(chain.from_iterable(
            [v, 0] for v in weight_decays)),
        change_with_lr=True
    )
    return optimizer


def prepare_model_bertadam(layers, learning_rates, weight_decays, start_layer=0):
    optimizer_fn = partial(
        BertAdam,
        params=layers[0].parameters(),
        lr=2e-5,
        warmup=0.05,
        weight_decay=0.01
    )
    return optimizer_fn


def prepare_bert_model(model, encoder_layer_target, base_lr, lr_decay, weight_decay, train_embeddings=False, new_param_only=False):
    if new_param_only:
        layers = [model.head]
        freeze_layers([model.bert] + layers, [True, False])
        learning_rates = [base_lr]
    else:
        layers = [
            model.bert.embeddings
        ] if encoder_layer_target < 0 or train_embeddings else []
        for i in range(max(0, encoder_layer_target), len(model.bert.encoder.layer), 4):
            layers.append(model.bert.encoder.layer[i:(i+4)])
        layers.append(model.head)
        freeze_layers([model.bert] + layers, [True] + [False] * len(layers))
        learning_rates = [
            base_lr * (lr_decay ** i)
            for i in range(len(layers)-1, -1, -1)
        ]
    return prepare_model(layers, learning_rates, [weight_decay] * len(layers))


def prepare_gpt2_model(model, encoder_layer_target, base_lr, lr_decay, weight_decay, train_embeddings=False, new_param_only=False):
    if new_param_only:
        layers = [model.head]
        freeze_layers([model.transformer] + layers, [True, False])
        learning_rates = [base_lr]
    else:
        layers = [
            nn.ModuleList([model.transformer.wte, model.transformer.wpe])
        ] if encoder_layer_target < 0 or train_embeddings else []
        for i in range(max(0, encoder_layer_target), len(model.transformer.h), 4):
            layers.append(model.transformer.h[i:(i+4)])
        layers.append(nn.ModuleList([model.head, model.transformer.ln_f]))
        freeze_layers([model.transformer] + layers,
                      [True] + [False] * len(layers))
        learning_rates = [
            base_lr * (lr_decay ** i)
            for i in range(len(layers)-1, -1, -1)
        ]
    return prepare_model(layers, learning_rates, [weight_decay] * len(layers))


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Bert Finetuning")
def main(args):
    if args.model.startswith("bert"):
        tokenizer = BertTokenizer.from_pretrained(
            args.model,
            do_lower_case=args.model.split("-")[-1] == "uncased",
            # never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        )
        model = get_bert_model(args.model).to(DEVICE)
        optimizer = prepare_bert_model(
            model, encoder_layer_target=args.start_layer,
            base_lr=1e-3, lr_decay=0.5623,
            weight_decay=args.weight_decay, train_embeddings=args.train_embd,
            new_param_only=True
        )
    elif args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.set_special_tokens(SPECIAL_TOKENS)
        # print("<cls>", tokenizer.convert_tokens_to_ids(["<cls>"]))
        model = get_gpt2_model(tokenizer).to(DEVICE)
        optimizer = prepare_gpt2_model(
            model, encoder_layer_target=args.start_layer,
            base_lr=1e-3, lr_decay=0.5623,
            weight_decay=args.weight_decay, train_embeddings=args.train_embd,
            new_param_only=True
        )
    else:
        raise ValueError("Unrecognized model.")
    joblib.dump(tokenizer, str(MODEL_DIR / f"{args.model}_tokenizer.jbl"))
    train_loader, valid_loader, df_valid = get_loaders(args, tokenizer)
    gc.collect()

    n_steps = len(train_loader) // 2
    # optimizer = optimizer(t_total=n_steps)
    use_amp = False
    if args.amp and APEX_AVAILABLE:
        use_amp = True
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.amp
        )

    bot = ToxicBot(
        checkpoint_dir=CACHE_DIR / "model_cache/",
        log_dir=Path(".") if ON_KAGGLE else CACHE_DIR / "logs/",
        model=model, train_loader=train_loader,
        val_loader=valid_loader, clip_grad=1.,
        optimizer=optimizer, echo=not ON_KAGGLE,
        criterion=WeightedBCELossWithLogit(),
        # WeightedFocalLoss(alpha=args.alpha, gamma=args.gamma),
        avg_window=len(train_loader) // 8,
        callbacks=[
            LearningRateSchedulerCallback(
                TriangularLR(
                    optimizer, 100, ratio=3, steps_per_cycle=n_steps)
                # GradualWarmupScheduler(
                #     bot.optimizer, 100, n_steps * 0.1,
                #     after_scheduler=CosineAnnealingLR(
                #         optimizer, n_steps * 0.9
                #     )
                # )
            )
        ],
        pbar=not ON_KAGGLE, use_tensorboard=False,
        device=DEVICE,
        use_amp=use_amp,
        gradient_accumulation_steps=args.grad_accu
    )
    bot.metrics.append(TargetFocalLoss(bot.criterion))
    bot.metrics.append(AuxFocalLoss(bot.criterion))
    bot.metrics.append(ToxicMetric(df_valid))
    bot.logger.info(bot.criterion)
    bot.logger.info("Start training head...")
    bot.train(
        n_steps,
        log_interval=len(train_loader) // 12,
        snapshot_interval=len(train_loader) // 2,
        keep_n_snapshots=1
    )
    bot.logger.info("Start fine-tuning...")
    if use_amp:
        if args.model.startswith("bert"):
            model = get_bert_model(args.model).to(DEVICE)
        else:
            model = get_gpt2_model(tokenizer).to(DEVICE)
        model.load_state_dict(bot.model.cpu().state_dict())
    if args.model.startswith("bert"):
        bot.optimizer = prepare_bert_model(
            model, encoder_layer_target=args.start_layer,
            base_lr=args.base_lr, lr_decay=args.lr_decay,
            weight_decay=args.weight_decay,
            train_embeddings=args.train_embd,
            new_param_only=False
        )
    else:
        bot.optimizer = prepare_gpt2_model(
            model, encoder_layer_target=args.start_layer,
            base_lr=args.base_lr, lr_decay=args.lr_decay,
            weight_decay=args.weight_decay,
            train_embeddings=args.train_embd,
            new_param_only=False
        )
    n_steps = len(train_loader) * args.epochs
    # bot.optimizer = bot.optimizer(t_total=n_steps)
    if use_amp:
        bot.model, bot.optimizer = amp.initialize(
            model, bot.optimizer, opt_level=args.amp
        )
    bot.count_model_parameters()
    bot.step = 0
    bot.callbacks = [
        LearningRateSchedulerCallback(
            # TriangularLR(
            #     bot.optimizer, 1000, ratio=9, steps_per_cycle=n_steps
            # )
            GradualWarmupScheduler(
                bot.optimizer, 100, int(n_steps * 0.1),
                after_scheduler=CosineAnnealingLR(
                    bot.optimizer, int(n_steps * 0.9)
                )
            )
        )
    ]
    bot.train(
        n_steps,
        log_interval=len(train_loader) // 12,
        snapshot_interval=len(train_loader) // 4,
        keep_n_snapshots=1
    )
    bot.load_model(bot.best_performers[0][1])
    bot.remove_checkpoints(keep=0)
    if use_amp:
        if args.model.startswith("bert"):
            model = get_bert_model(args.model).cpu()
        else:
            model = get_gpt2_model(tokenizer).cpu()
        model.load_state_dict(bot.model.cpu().state_dict())
        torch.save(model, str(
            MODEL_DIR / f"{args.model}_{args.sample_size}_{args.weight_config}_{args.maxlen}_f{args.fold}.pth"))
        # torch.save(bot.model.state_dict(), str(
        #     MODEL_DIR / f"{BERT_MODEL}.pth"))
        del model
    else:
        torch.save(bot.model, str(
            MODEL_DIR / f"{args.model}_{args.sample_size}_{args.weight_config}_{args.maxlen}_f{args.fold}.pth"))
        # torch.save(bot.model.state_dict(), str(
        #     MODEL_DIR / f"{BERT_MODEL}.pth"))


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
