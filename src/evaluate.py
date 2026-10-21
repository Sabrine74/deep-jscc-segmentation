"""
evaluate.py — mIoU as a function of the test SNR: the main figure of the project.

    python evaluate.py --ckpt runs/exp/best.pt --data camvid --root ../data/CamVid

For each test SNR, the channel is applied --reps times (the noise is random)
and the confusion matrix is accumulated.
Outputs: <run>/miou_vs_snr.csv and <run>/miou_vs_snr.png
"""
import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from data import CAMVID_IGNORE, CamVid, SyntheticRoad
from metrics import SegMetric
from model import DeepJSCC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", choices=["synthetic", "camvid"], default="synthetic")
    p.add_argument("--root", default="../data/CamVid")
    p.add_argument("--channel", default=None, help="test channel (default: training channel)")
    p.add_argument("--snrs", default="-5,-2,0,2,4,6,8,10,12,15,20")
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device)
    ar = ck["args"]
    model = DeepJSCC(c=ar["c"], num_classes=ck["K"], task="seg",
                     channel=a.channel or ar["channel"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if a.data == "synthetic":
        ds, ignore = SyntheticRoad(256, seed=3), None
    else:
        ds, ignore = CamVid(a.root, "test"), CAMVID_IGNORE
    dl = DataLoader(ds, 8, shuffle=False)
    rows = []
    with torch.no_grad():
        for s in [float(v) for v in a.snrs.split(",")]:
            m = SegMetric(ck["K"], ignore)
            for _ in range(a.reps):
                for x, y in dl:
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x, torch.full((x.shape[0], 1), s, device=device))
                    m.update(logits, y)
            rows.append((s, m.miou(), m.pixel_acc()))
            print(f"SNR = {s:5.1f} dB   mIoU = {rows[-1][1]:.4f}   pixel acc. = {rows[-1][2]:.4f}")
    out = os.path.dirname(a.ckpt)
    with open(os.path.join(out, "miou_vs_snr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "miou", "pixel_acc"])
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.5, 3))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color="#5DAA78")
    plt.xlabel("test SNR (dB)")
    plt.ylabel("mIoU")
    plt.grid(alpha=0.3)
    plt.title(f"Deep JSCC, {model.channel.kind} channel, c = {ar['c']}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "miou_vs_snr.png"), dpi=200)


if __name__ == "__main__":
    main()
