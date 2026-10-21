"""
train.py — End-to-end training of the segmentation-oriented Deep JSCC.

Examples (run from the src/ folder):
    # quick check without data (CPU, a few minutes)
    python train.py --data synthetic --epochs 5 --c 8

    # CamVid, AWGN channel, training SNR drawn uniformly in [0, 20] dB
    python train.py --data camvid --root ../data/CamVid --c 8 --epochs 150 --channel awgn

    # same on block Rayleigh fading
    python train.py --data camvid --root ../data/CamVid --c 8 --epochs 150 --channel rayleigh
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CAMVID_IGNORE, CamVid, SyntheticRoad
from metrics import SegMetric
from model import DeepJSCC


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "camvid"], default="synthetic")
    p.add_argument("--root", default="../data/CamVid")
    p.add_argument("--c", type=int, default=16, help="transmitted channels (k/n = c/96)")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--snr_min", type=float, default=0.0)
    p.add_argument("--snr_max", type=float, default=20.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_data(a):
    if a.data == "synthetic":
        return SyntheticRoad(512, seed=1), SyntheticRoad(128, seed=2), SyntheticRoad.NUM_CLASSES, None
    return CamVid(a.root, "train", augment=True), CamVid(a.root, "val"), 11, CAMVID_IGNORE


@torch.no_grad()
def evaluate(model, loader, K, ignore, snr_db, device):
    model.eval()
    m = SegMetric(K, ignore)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, torch.full((x.shape[0], 1), float(snr_db), device=device))
        m.update(logits, y)
    return m.miou()


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, K, ignore = build_data(a)
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(va, a.bs, shuffle=False, num_workers=2)
    model = DeepJSCC(c=a.c, num_classes=K, task="seg", channel=a.channel,
                     snr_range=(a.snr_min, a.snr_max)).to(device)
    H, W = tr[0][0].shape[1:]
    print(f"device={device}  image={H}x{W}  k/n={model.bandwidth_ratio(H, W):.4f}  classes={K}")
    crit = nn.CrossEntropyLoss(ignore_index=ignore if ignore is not None else -100)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    os.makedirs(a.out, exist_ok=True)
    hist, best = [], -1
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)            # SNR drawn at random in [snr_min, snr_max]
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()                 # the gradient flows through the channel
            opt.step()
            tot += loss.item() * x.shape[0]
        sched.step()
        tr_loss = tot / len(tl.dataset)
        mi = {s: evaluate(model, vl, K, ignore, s, device) for s in (0, 10, 20)}
        hist.append(dict(epoch=ep, loss=tr_loss, **{f"miou_{s}dB": v for s, v in mi.items()}))
        print(f"epoch {ep:3d}  loss={tr_loss:.3f}  mIoU@0/10/20 dB = "
              f"{mi[0]:.3f}/{mi[10]:.3f}/{mi[20]:.3f}  ({time.time() - t0:.0f}s)")
        if mi[10] > best:
            best = mi[10]
            torch.save(dict(model=model.state_dict(), args=vars(a), K=K), os.path.join(a.out, "best.pt"))
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    print("best validation mIoU at 10 dB:", round(best, 4))


if __name__ == "__main__":
    main()
