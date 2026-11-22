"""Watershed JEPA: context encoder E on the periodic cylinder (window zeroed, mask channel), predictor P
mapping E's latent to the watershed summary of the masked window (basin count, top-8 lifetimes, top-8
per-basin water). Targets come from hydraulics.window_targets on the FULL terrain (never learned)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 8
NCLS = 8          # basin count 0..7 (7 = ">= 7"), class index = count
WIN_R, WIN_C = 8, 16


class CylConv(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=0)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")   # phi periodic
        x = F.pad(x, (0, 0, 1, 1))                    # eta: zero outside the cylinder
        return self.conv(x)


class WSEncoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.convs = nn.ModuleList([CylConv(2, 64, 1), CylConv(64, 128, 2),
                                    CylConv(128, 256, 2), CylConv(256, 512, 2)])
        self.fc = nn.Linear(512 * 4 * 8, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, h, win):
        """h (B,32,64) float terrain, win (B,32,64) bool mask of the hidden window (zeroed)."""
        x = torch.stack([h.masked_fill(win, 0.0) / 4.0, win.float()], 1)
        for c in self.convs:
            x = F.gelu(c(x))
        return self.ln(self.fc(x.flatten(1)))


class WSJEPA(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.enc = WSEncoder(d)
        self.pred = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, NCLS + 2 * K))

    def heads(self, z):
        o = self.pred(z)
        return o[:, :NCLS], o[:, NCLS:NCLS + K], o[:, NCLS + K:]


def sample_windows(B, device, gen=None):
    """Random 8 x 16 blocks: rows contiguous (no wrap), phi columns wrap."""
    r0 = torch.randint(0, 32 - WIN_R + 1, (B,), device=device, generator=gen)
    c0 = torch.randint(0, 64, (B,), device=device, generator=gen)
    rows = torch.arange(32, device=device)
    cols = torch.arange(64, device=device)
    rm = (rows[None] >= r0[:, None]) & (rows[None] < r0[:, None] + WIN_R)
    cm = ((cols[None] - c0[:, None]) % 64) < WIN_C
    return rm[:, :, None] & cm[:, None, :]


def loss_terms(model, h, lab, life, win):
    """h (B,32,64) float, lab (B,2048) long, life (B,2048) float, win (B,32,64) bool -> dict of losses."""
    from hydraulics import window_targets
    B = h.shape[0]
    tg = window_targets(h.reshape(B, -1), lab, life, win.reshape(B, -1))
    z = model.enc(h, win)
    logits, pers_p, water_p = model.heads(z)
    cls = tg["count"].clamp(0, NCLS - 1)
    ce = F.cross_entropy(logits, cls)
    pers_t = tg["pers"].float()
    water_t = torch.log1p(tg["water"].float())
    l_p = F.smooth_l1_loss(pers_p, pers_t)
    l_w = F.smooth_l1_loss(water_p, water_t)
    return {"loss": ce + l_p + l_w, "ce": ce, "pers": l_p, "water": l_w,
            "acc": (logits.argmax(1) == cls).float().mean(),
            "w3_err": (tg["ledger_total"] - tg["water_total"]).abs().max()}
