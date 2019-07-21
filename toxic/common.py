import os
from dataclasses import dataclass
from pathlib import Path

import torch
from helperbot import BaseBot, AUC as BaseAUC, Metric

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ
DATA_DIR = Path(
    "../input/jigsaw-unintended-bias-in-toxicity-classification/"
) if ON_KAGGLE else Path("data/")
CACHE_DIR = Path("/tmp/") if ON_KAGGLE else Path("data/cache/")
MODEL_DIR = Path(".") if ON_KAGGLE else CACHE_DIR


class AUC(BaseAUC):
    def __call__(self, truth: torch.Tensor, pred: torch.Tensor):
        return super().__call__(truth[:, 1], pred[:, 0])


class AuxFocalLoss(Metric):
    """AUC for binary targets"""
    name = "aux_focal"

    def __init__(self, focal_loss):
        super().__init__()
        self.focal_loss = focal_loss

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor):
        _, aux_loss = self.focal_loss(pred, truth, split_returns=True)
        return aux_loss, f"{aux_loss:.6f}"


class TargetFocalLoss(Metric):
    """AUC for binary targets"""
    name = "target_focal"

    def __init__(self, focal_loss):
        super().__init__()
        self.focal_loss = focal_loss

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor):
        target_loss, _ = self.focal_loss(pred, truth, split_returns=True)
        return target_loss, f"{target_loss:.6f}"


@dataclass
class ToxicBot(BaseBot):
    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"
        self.metrics = []
        self.monitor_metric = "toxic_auc"

    def extract_prediction(self, x):
        return x


def tokenize(row, tokenizer):
    # Saved for more text cleanings
    return tokenizer.tokenize(row.comment_text)
