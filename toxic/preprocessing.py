import torch
import numpy as np
import pandas as pd

IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


def custom_data_weights(df):
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    df_identities = df[IDENTITY_COLUMNS].fillna(0)
    df_identities = (df_identities >= 0.5).astype(np.float32)
    identity_cnts = df_identities.sum(axis=0).values
    identity_ratio = np.sum(identity_cnts) / identity_cnts
    print(identity_ratio)
    # setting weights
    df["weight"] = np.sum(df_identities.values *
                          identity_ratio[None, :], axis=1)
    df["weight"] = df["weight"] / np.max(identity_ratio) * 4 + .5
    # Background positive
    df["weight"] += (
        df["target"].values *
        (df_identities.sum(axis=1).values == 0)
    ) * .5
    # Subgroup positive
    df["weight"] += (
        df["target"].values *
        df_identities.sum(axis=1).values
    ) * .1
    df["weight"] = df["weight"] / df["weight"].mean()
    print(df["weight"].describe())
    return df


def kaggle_data_weights(df):
    # Reference: https://www.kaggle.com/mashhoori/a-loss-function-for-the-jigsaw-competition
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    df_identities = df[IDENTITY_COLUMNS].fillna(0)
    df_identities = (df_identities >= 0.5).astype(np.uint8)
    df["target"] = df.target.astype(bool)
    # Overall
    df["weight"] = 1.
    df.loc[df.target, "weight"] = 1 / df.target.sum()
    df.loc[~df.target, "weight"] = 1 / (~df.target).sum()
    # Subgroup
    for col in IDENTITY_COLUMNS:
        hasIdentity = df[col].astype(bool)
        # These samples participate in the subgroup AUC and BNSP terms
        df.loc[hasIdentity & df['target'], "weight"] += (
            2 / ((hasIdentity & df['target']).sum() * len(IDENTITY_COLUMNS)))
        # These samples participate in the subgroup AUC and BPSN terms
        df.loc[hasIdentity & ~df['target'], "weight"] += (
            2 / ((hasIdentity & ~df['target']).sum() * len(IDENTITY_COLUMNS)))
        # These samples participate in the BPSN term
        df.loc[~hasIdentity & df['target'], "weight"] += (
            1 / ((~hasIdentity & df['target']).sum() * len(IDENTITY_COLUMNS)))
        # These samples participate in the BNSP term
        df.loc[~hasIdentity & ~df['target'], "weight"] += (
            1 / ((~hasIdentity & ~df['target']).sum() * len(IDENTITY_COLUMNS)))
    df["weight"] = df["weight"] / df["weight"].mean()
    print(df["weight"].describe())
    df["target"] = df.target.astype(np.float32)
    return df


def yuval_weights(df):
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    # https://www.kaggle.com/yuval6967/toxic-train-bert-base-pytorch
    has_identity = torch.sigmoid(
        10*(torch.tensor((df[IDENTITY_COLUMNS].fillna(0).max(axis=1)).values)-0.4))
    has_target = torch.sigmoid(
        10*(torch.tensor(df['target_raw'].values)-0.4))
    weights = (torch.ones(df.shape[0], dtype=torch.float64)+has_identity +
               has_identity*(1-has_target)+has_target*(1-has_identity)) / 4
    df["weight"] = weights.to(dtype=torch.float32).numpy()
    df["weight"] = df["weight"] / df["weight"].mean()
    print(df["weight"].describe())
    return df
