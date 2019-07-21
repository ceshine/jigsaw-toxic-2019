from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from helperbot.metrics import Metric

from .preprocessing import IDENTITY_COLUMNS

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive


def calculate_overall_auc(df, model_name):
    true_labels = df["target"]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(
        non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(
        non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(
            dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(
            dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(
            dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)


def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + IDENTITY_COLUMNS:
        convert_to_bool(bool_df, col)
    return bool_df


class ToxicMetric(Metric):
    name = "toxic_auc"

    def __init__(self, df_valid):
        super().__init__()
        self.df_valid = convert_dataframe_to_bool(df_valid)

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        valid_pred = torch.sigmoid(pred).cpu().numpy()[:, 0]
        self.df_valid["pred"] = valid_pred
        df_bias_metrics = compute_bias_metrics_for_model(
            self.df_valid, IDENTITY_COLUMNS, 'pred', 'target').fillna(0)
        # more options can be specified also
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_bias_metrics)
        overall_auc = calculate_overall_auc(self.df_valid, "pred")
        print(f"Overall AUC: {overall_auc:.6f}")
        print(
            f"Mean bnsp_auc: {power_mean(df_bias_metrics['bnsp_auc'], -5):.6f}")
        print(
            f"Mean bpsn_auc: {power_mean(df_bias_metrics['bpsn_auc'], -5):.6f}")
        print(
            f"Mean subgroup auc: {power_mean(df_bias_metrics['subgroup_auc'], -5):.6f}")
        final = get_final_metric(df_bias_metrics, overall_auc)
        print(f"Final score: {final:.6f}")
        return final * -1, f"{final * 100:.2f}"
