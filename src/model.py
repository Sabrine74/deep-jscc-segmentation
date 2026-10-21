"""
model.py — Task-oriented Deep JSCC for semantic segmentation.

    image x (B,3,H,W)
      -> CNN encoder f_theta -> z (B,c,H/4,W/4) -> power normalization
                                                -> channel (AWGN / Rayleigh)
      <- CNN decoder g_phi   <- z_hat (B,c,H/4,W/4)
      -> segmentation logits (B,K,H,W)   [task="seg"]
      -> reconstructed image (B,3,H,W)   [task="recon", reconstruction baseline]

Bandwidth ratio: k/n = (c * H/4 * W/4 / 2) / (3 * H * W) = c / 96
    c = 8 -> k/n = 1/12        c = 16 -> k/n = 1/6
"""
import torch
import torch.nn as nn

from channel import Channel, power_normalize, to_complex, to_real


def conv(cin, cout, k=5, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2), nn.BatchNorm2d(cout), nn.PReLU(cout))


def deconv(cin, cout, k=5, s=2):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, k, s, k // 2, output_padding=s - 1),
                         nn.BatchNorm2d(cout), nn.PReLU(cout))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(conv(c, c, 3), nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c))
        self.act = nn.PReLU(c)

    def forward(self, x):
        return self.act(x + self.b(x))  # residual connection (He et al., 2016)


class Encoder(nn.Module):
    """3 x H x W  ->  c x H/4 x W/4  (c = number of transmitted channels)."""

    def __init__(self, c=16, width=128):
        super().__init__()
        self.net = nn.Sequential(
            conv(3, width // 2, 5, 2),      # H/2
            conv(width // 2, width, 5, 2),  # H/4
            ResBlock(width), ResBlock(width),
            nn.Conv2d(width, c, 3, 1, 1),   # no activation: symbols can take any value
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """c x H/4 x W/4 (+ SNR map)  ->  out_ch x H x W."""

    def __init__(self, c=16, out_ch=12, width=128, snr_cond=True):
        super().__init__()
        self.snr_cond = snr_cond
        cin = c + (1 if snr_cond else 0)
        self.net = nn.Sequential(
            conv(cin, width, 3),
            ResBlock(width), ResBlock(width),
            deconv(width, width // 2),       # H/2
            deconv(width // 2, width // 2),  # H
            nn.Conv2d(width // 2, out_ch, 3, 1, 1),
        )

    def forward(self, z_hat, snr_db):
        if self.snr_cond:
            B, _, h, w = z_hat.shape
            snr_map = (snr_db.reshape(B, 1, 1, 1) / 20.0).expand(B, 1, h, w)
            z_hat = torch.cat([z_hat, snr_map], 1)
        return self.net(z_hat)


class DeepJSCC(nn.Module):
    """task="seg": output = logits (K classes); task="recon": output = image in [0, 1]."""

    def __init__(self, c=16, num_classes=12, task="seg", channel="awgn",
                 snr_range=(0.0, 20.0), snr_cond=True):
        super().__init__()
        self.task = task
        self.encoder = Encoder(c)
        out_ch = num_classes if task == "seg" else 3
        self.decoder = Decoder(c, out_ch, snr_cond=snr_cond)
        self.channel = Channel(channel, snr_range)

    def bandwidth_ratio(self, H, W):
        with torch.no_grad():
            z = self.encoder(torch.zeros(1, 3, H, W, device=next(self.parameters()).device))
        return (z.numel() / 2) / (3 * H * W)

    def forward(self, x, snr_db=None):
        z = self.encoder(x)                     # (B, c, H/4, W/4) real
        zc = power_normalize(to_complex(z))     # (B, k) complex, P = 1
        yc, snr_db = self.channel(zc, snr_db)   # noise (+ fading)
        z_hat = to_real(yc, z.shape)            # (B, c, H/4, W/4)
        out = self.decoder(z_hat, snr_db)
        if self.task == "recon":
            out = torch.sigmoid(out)
        return out, snr_db


if __name__ == "__main__":
    m = DeepJSCC(c=16, num_classes=12)
    x = torch.rand(2, 3, 256, 512)
    y, _ = m(x)
    print("input:", tuple(x.shape), "-> output:", tuple(y.shape))
    r = m.bandwidth_ratio(256, 512)
    print("k/n = %.4f (= 1/%.0f)" % (r, 1 / r))
    print("parameters: %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
