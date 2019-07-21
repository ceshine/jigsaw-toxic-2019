import torch
import torch.nn as nn
import torch.nn.functional as F

# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109


class WeightedBCELossWithLogit(nn.Module):
    def forward(self, preds, targets, split_returns=False):
        weights, truths = targets[:, 0], targets[:, 1:]
        weights = weights.unsqueeze(-1).repeat(1, truths.size(1))
        weights[:, 0] = weights[:, 0]*(truths.size(1))/4
        weights[:, 6:] = weights[:, 6:]*0.25
        if split_returns:
            target_loss = F.binary_cross_entropy_with_logits(
                preds[:, :1], truths[:, :1],
                weight=weights[:, :1], reduction="none"
            )
            aux_loss_toxic = F.binary_cross_entropy_with_logits(
                preds[:, 1:6], truths[:, 1:6],
                weight=weights[:, 1:6], reduction="none"
            )
            aux_loss_ident = F.binary_cross_entropy_with_logits(
                preds[:, 6:], truths[:, 6:],
                weight=weights[:, 6:], reduction="none"
            )
            return torch.mean(
                target_loss
            ) / 15, torch.mean(
                torch.cat([
                    aux_loss_toxic,
                    aux_loss_ident
                ], dim=1)
            ) * 14 / 15
        return F.binary_cross_entropy_with_logits(
            preds, truths, weight=weights, reduction="mean"
        )


class FocalLoss(nn.Module):
    """Adapted from: https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
    F.logsimoid used as in https://gist.github.com/AdrienLE/bf31dfe94569319f6e47b2de8df13416#file-focal_dice_1-py
    """

    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        assert alpha > 0 and alpha < 1
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        y = y.float()
        pt_log = F.logsigmoid(-x * (y * 2 - 1))
        # w = alpha if t > 0 else 1-alpha
        at = (self.alpha * y + (1-self.alpha) * (1-y)) * 2
        w = at * (pt_log * self.gamma).exp()
        # Don't calculate gradients of the weights
        w = w.detach()
        return F.binary_cross_entropy_with_logits(x, y, w, reduction="mean")

    def __str__(self):
        return f"<Focal Loss alpha={self.alpha} gamma={self.gamma}>"


class WeightedFocalLoss(FocalLoss):
    def forward(self, preds, targets, split_returns=False):
        weights, truths = targets[:, 0], targets[:, 1:]
        y = truths.float()
        pt_log = F.logsigmoid(-preds * (y * 2 - 1))
        # w = alpha if t > 0 else 1-alpha
        at = (self.alpha * y + (1-self.alpha) * (1-y)) * 2
        # w = at * (pt_log * self.gamma).exp() * torch.cat([
        #     weights.unsqueeze(1),
        #     torch.ones(
        #         weights.size(0), y.size(1)-1,
        #         dtype=torch.float32
        #     ).to(weights.device)
        # ], dim=1)
        w = at * (pt_log * self.gamma).exp() * weights.unsqueeze(1)
        # Don't calculate gradients of the weights
        w = w.detach()
        loss = torch.mean(
            F.binary_cross_entropy_with_logits(
                preds, y, w, reduction="none"
            ), dim=0)
        if split_returns:
            return loss[0], torch.mean(loss[1:])
        return loss[0] + torch.mean(loss[1:])


class FbetaLoss(nn.Module):
    def __init__(self, beta=1):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / \
            (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss


def test_focal_loss():
    import numpy as np
    # case 1
    probs = np.array([.8, .2])
    logits = np.log(probs / (1-probs))
    targets = np.array([1, 0])
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    loss = FocalLoss(alpha=.75, gamma=1.)(
        logits_tensor, targets_tensor
    ).item()
    print(loss)
    assert np.isclose(loss, .02231435 * 2)
    # case 2
    probs = np.array([.8, .1, .5])
    logits = np.log(probs / (1-probs))
    targets = np.array([1, 0, 0])
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    loss = FocalLoss(alpha=.8, gamma=.5)(
        logits_tensor, targets_tensor
    ).item()
    print(loss)
    assert np.isclose(loss, 0.0615079 * 2)
    # case 3
    loss = FocalLoss(alpha=.5, gamma=0)(
        logits_tensor, targets_tensor
    ).item()
    print(loss)
    assert np.isclose(
        F.binary_cross_entropy_with_logits(
            logits_tensor, targets_tensor).item(),
        loss
    )


if __name__ == "__main__":
    test_focal_loss()
