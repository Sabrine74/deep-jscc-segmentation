"""
channel.py — Differentiable wireless channel layer for Deep JSCC (PyTorch).

Convention: the real encoder output z of shape (B, C, H, W) is read as a block
of k = C*H*W/2 complex symbols (first half of the channels = real part,
second half = imaginary part). One image = one block of k channel uses.

Every operation is differentiable, so the gradient of the task loss flows
through the channel back to the encoder: this is what makes end-to-end
training possible.
"""
import torch
import torch.nn as nn


def to_complex(z: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) real, C even  ->  (B, k) complex, k = C*H*W/2."""
    B, C, H, W = z.shape
    assert C % 2 == 0, "the number of encoder output channels must be even"
    re, im = z[:, : C // 2], z[:, C // 2:]
    return torch.complex(re.reshape(B, -1), im.reshape(B, -1))


def to_real(zc: torch.Tensor, shape) -> torch.Tensor:
    """Inverse of to_complex."""
    B, C, H, W = shape
    re = zc.real.reshape(B, C // 2, H, W)
    im = zc.imag.reshape(B, C // 2, H, W)
    return torch.cat([re, im], dim=1)


def power_normalize(zc: torch.Tensor, P: float = 1.0) -> torch.Tensor:
    """Enforce the average power constraint (1/k) * sum |z_i|^2 = P, per image.

    Without it, the network would simply increase the amplitude of its
    symbols to drown the noise (infinite effective SNR).
    """
    k = zc.shape[1]
    norm = torch.linalg.vector_norm(zc, dim=1, keepdim=True)  # ||z||_2 per image
    return torch.sqrt(torch.tensor(k * P, device=zc.device)) * zc / (norm + 1e-8)


def complex_noise(shape, sigma2, device):
    """CN(0, sigma2) noise: real and imaginary parts each have variance sigma2/2."""
    s = torch.sqrt(sigma2 / 2)
    return torch.complex(torch.randn(shape, device=device), torch.randn(shape, device=device)) * s


class Channel(nn.Module):
    """AWGN or block-Rayleigh-fading channel (perfect CSI at the receiver).

    Args:
        kind: "awgn" | "rayleigh"
        snr_range: (min, max) in dB. During training one SNR is drawn
            uniformly for EACH image, so the model learns to be robust.
        P: average power per complex symbol.
    """

    def __init__(self, kind="awgn", snr_range=(0.0, 20.0), P=1.0):
        super().__init__()
        assert kind in ("awgn", "rayleigh")
        self.kind, self.snr_range, self.P = kind, snr_range, P

    def sample_snr(self, B, device):
        lo, hi = self.snr_range
        return torch.empty(B, 1, device=device).uniform_(lo, hi)

    def forward(self, zc: torch.Tensor, snr_db: torch.Tensor = None):
        """zc: (B, k) complex, already power-normalized.
        snr_db: (B, 1) in dB, or None (random draw).
        Returns (equalized received symbols, snr_db)."""
        B = zc.shape[0]
        if snr_db is None:
            snr_db = self.sample_snr(B, zc.device)
        snr_db = snr_db.reshape(B, 1).to(zc.device)
        sigma2 = self.P / (10.0 ** (snr_db / 10.0))  # SNR = P / sigma^2
        n = complex_noise(zc.shape, sigma2, zc.device)
        if self.kind == "awgn":
            y = zc + n
        else:
            # one coefficient h ~ CN(0, 1) per image (slow, block fading)
            h = complex_noise((B, 1), torch.ones(B, 1, device=zc.device), zc.device)
            y = h * zc + n
            # MMSE equalization with perfect knowledge of h (CSI)
            y = torch.conj(h) * y / (h.abs() ** 2 + sigma2)
        return y, snr_db


if __name__ == "__main__":
    # Numerical sanity check: measured power and measured SNR.
    torch.manual_seed(0)
    z = torch.randn(4, 16, 32, 64) * 3.0 + 1.0
    zc = power_normalize(to_complex(z))
    print("average power after normalization:", round((zc.abs() ** 2).mean().item(), 4))
    ch = Channel("awgn")
    y, _ = ch(zc, torch.full((4, 1), 10.0))
    noise_p = ((y - zc).abs() ** 2).mean().item()
    print("measured SNR: %.2f dB (expected 10 dB)" % (10 * torch.log10(torch.tensor(1 / noise_p))).item())
    print("shape after to_real:", tuple(to_real(zc, z.shape).shape))
