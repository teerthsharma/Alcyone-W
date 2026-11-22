"""Encoder variant with 2 + 8 input channels (terrain, mask, 8 land channels). Same trunk, predictor and
loss as model.py (imported, not edited). Land channels are zeroed inside the hidden window like the terrain."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import CylConv, K, NCLS  # noqa: E402

N_LAND = 8


class LandEncoder(nn.Module):
    def __init__(self, d=256, n_land=N_LAND):
        super().__init__()
        cin = 2 + n_land
        self.convs = nn.ModuleList([CylConv(cin, 64, 1), CylConv(64, 128, 2),
                                    CylConv(128, 256, 2), CylConv(256, 512, 2)])
        self.fc = nn.Linear(512 * 4 * 8, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, h, win, land):
        """h (B,32,64) terrain, win (B,32,64) bool hidden window, land (B,8,32,64) land channels."""
        x = torch.cat([torch.stack([h.masked_fill(win, 0.0) / 4.0, win.float()], 1),
                       land.masked_fill(win[:, None], 0.0)], 1)
        for c in self.convs:
            x = F.gelu(c(x))
        return self.ln(self.fc(x.flatten(1)))


class LandJEPA(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.enc = LandEncoder(d)
        self.pred = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, NCLS + 2 * K))

    def heads(self, z):
        o = self.pred(z)
        return o[:, :NCLS], o[:, NCLS:NCLS + K], o[:, NCLS + K:]


def loss_land_terms(model, h, land, lab, life, win):
    """Identical to model.loss_terms (same targets, same loss weights), with the land channels added."""
    from hydraulics import window_targets
    B = h.shape[0]
    tg = window_targets(h.reshape(B, -1), lab, life, win.reshape(B, -1))
    z = model.enc(h, win, land)
    logits, pers_p, water_p = model.heads(z)
    cls = tg["count"].clamp(0, NCLS - 1)
    ce = F.cross_entropy(logits, cls)
    l_p = F.smooth_l1_loss(pers_p, tg["pers"].float())
    l_w = F.smooth_l1_loss(water_p, torch.log1p(tg["water"].float()))
    return {"loss": ce + l_p + l_w, "ce": ce, "pers": l_p, "water": l_w,
            "acc": (logits.argmax(1) == cls).float().mean(),
            "w3_err": (tg["ledger_total"] - tg["water_total"]).abs().max()}
