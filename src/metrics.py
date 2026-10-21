"""
metrics.py — mIoU, pixel accuracy, PSNR.

IoU_c = TP_c / (TP_c + FP_c + FN_c)        mIoU = mean of IoU_c over classes
A K x K confusion matrix is accumulated over the whole test set and the IoU
is computed at the end (not per image, which would bias the mean).
"""
import torch


class SegMetric:
    def __init__(self, num_classes, ignore_index=None):
        self.K = num_classes
        self.ignore = ignore_index
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1).flatten().cpu()
        t = target.flatten().cpu()
        keep = (t >= 0) & (t < self.K)
        if self.ignore is not None:
            keep &= t != self.ignore
        idx = t[keep] * self.K + pred[keep]
        self.cm += torch.bincount(idx, minlength=self.K ** 2).reshape(self.K, self.K)

    def iou(self):
        tp = self.cm.diag().float()
        fp = self.cm.sum(0).float() - tp
        fn = self.cm.sum(1).float() - tp
        denom = tp + fp + fn
        return torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))

    def miou(self):
        return torch.nanmean(self.iou()).item()

    def pixel_acc(self):
        return (self.cm.diag().sum() / self.cm.sum().clamp(min=1)).item()


def psnr(x, x_hat, max_val=1.0):
    mse = torch.mean((x - x_hat) ** 2, dim=(1, 2, 3))
    return (10 * torch.log10(max_val ** 2 / mse)).mean().item()
