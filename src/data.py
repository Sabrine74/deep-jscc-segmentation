"""
data.py — Datasets.

1) CamVid (layout of the SegNet-Tutorial repository by Alex Kendall):
       data/CamVid/train/*.png       data/CamVid/trainannot/*.png
       data/CamVid/val/*.png         data/CamVid/valannot/*.png
       data/CamVid/test/*.png        data/CamVid/testannot/*.png
   360x480 images, 11 classes (0..10) + 11 = "void" (ignored).
   367 train / 101 val / 233 test images.

2) SyntheticRoad: small toy road scenes generated on the fly (no download).
   Used ONLY to check that the pipeline runs end to end on a CPU.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

CAMVID_CLASSES = ["sky", "building", "pole", "road", "sidewalk", "tree",
                  "sign", "fence", "car", "pedestrian", "bicyclist"]
CAMVID_IGNORE = 11


class CamVid(Dataset):
    def __init__(self, root, split="train", size=(256, 352), augment=False):
        from PIL import Image
        self.Image = Image
        self.imgs = sorted(glob.glob(os.path.join(root, split, "*.png")))
        self.lbls = sorted(glob.glob(os.path.join(root, split + "annot", "*.png")))
        assert len(self.imgs) == len(self.lbls) > 0, f"CamVid not found in {root}/{split}"
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        H, W = self.size
        img = self.Image.open(self.imgs[i]).convert("RGB").resize((W, H), self.Image.BILINEAR)
        lbl = self.Image.open(self.lbls[i]).resize((W, H), self.Image.NEAREST)  # NEAREST for labels!
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(lbl, np.int64))
        if self.augment and torch.rand(1) < 0.5:  # horizontal flip
            x, y = x.flip(-1), y.flip(-1)
        return x, y


class SyntheticRoad(Dataset):
    """Toy road scenes, 7 classes (sky, building, road, sidewalk, vegetation, car, pedestrian)."""
    NUM_CLASSES = 7

    def __init__(self, n=512, size=(64, 128), seed=0):
        self.n, self.size, self.seed = n, size, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        H, W = self.size
        r = np.random.default_rng(self.seed * 100003 + i)
        yy, xx = np.mgrid[0:H, 0:W]
        m = np.zeros((H, W), np.int64)
        hz = int(H * r.uniform(0.38, 0.52))
        x0 = 0
        while x0 < W:
            w, h = r.integers(W // 14, W // 6), r.integers(H // 8, H // 3)
            m[(yy > hz - h) & (yy <= hz) & (xx >= x0) & (xx < x0 + w - 1)] = 1
            x0 += w
        m[yy > hz] = 3
        vx = W / 2 + r.uniform(-W / 6, W / 6)
        m[(yy > hz) & (np.abs(xx - vx) < (yy - hz) * r.uniform(1.2, 2.0) + 3)] = 2
        for _ in range(r.integers(0, 3)):
            cx, cy = r.uniform(0, W), hz - r.uniform(2, H / 8)
            m[((xx - cx) ** 2 / (W / 18) ** 2 + (yy - cy) ** 2 / (H / 10) ** 2) < 1] = 4
        for _ in range(r.integers(1, 3)):
            cx, cy = vx + r.uniform(-W / 5, W / 5), hz + r.uniform(H / 8, H / 3)
            s = (cy - hz) / (H / 3) + 0.3
            m[(np.abs(xx - cx) < 10 * s) & (np.abs(yy - cy) < 5 * s)] = 5
        for _ in range(r.integers(0, 3)):
            px, py = r.uniform(0, W), hz + r.uniform(2, H / 3)
            m[(np.abs(xx - px) < 1.5) & (yy > py - 8) & (yy < py)] = 6
        base = np.array([[150, 195, 235], [160, 140, 150], [95, 95, 105], [190, 170, 140],
                         [80, 140, 80], [200, 60, 60], [230, 170, 80]], np.float32)
        img = base[m] * r.uniform(0.8, 1.2) + r.normal(0, 10, (H, W, 3))
        x = torch.from_numpy(np.clip(img, 0, 255).astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.from_numpy(m)
