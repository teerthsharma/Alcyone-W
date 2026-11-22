"""Alcyone watershed JEPA with a Galactica-6.7B warp land: the public API.

An event = up to 15 object slots, float32 (15, 8): [pT GeV, eta, phi, mass, charge, type, btag, valid];
type 0 muon, 1 electron, 2 jet, 3 MET (MET is not deposited). Pipeline:
  1. terrain h = -log(1 + pT deposit) on the 32 x 64 (eta, phi) cylinder (terrain.py);
  2. warp land: each visible object gets a caption ("muon, negative charge, pT 40 GeV, eta 1.0"); frozen
     Galactica-6.7B embeds it (layer 24, mean-pooled), a fixed PCA maps it to 8 numbers, and those are
     deposited on the cylinder with the same kernel (land_terrain.py). The 6,936 captions of the training
     data are pre-embedded in land/psi_gal6.7b_L24.npy; other captions need Galactica (loaded from the hub).
  3. the JEPA encoder sees terrain + land with a masked window and predicts the window's watershed summary
     (basin count, H0 persistence, water per basin); exact_summary() computes the true summary by the
     hydraulics (union-find flood, elder rule), never learned.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from terrain import build_terrain  # noqa: E402
from land_terrain import build_land  # noqa: E402
from model import sample_windows, K, NCLS, WIN_R, WIN_C  # noqa: E402
from model_land import LandJEPA  # noqa: E402

ROOT = os.path.dirname(HERE)
TYPE_NAME = {0: "muon", 1: "electron", 2: "jet"}
SIGN_NAME = {-1: "negative", 0: "neutral", 1: "positive"}
TAU = -float(np.log(2.0))           # basins = components of {deposit >= 1 GeV}
GALACTICA = "facebook/galactica-6.7b"
LAYER, MAX_LEN = 24, 96


def caption_records(x):
    """x (N,15,8) -> (N,15,4) int: type, charge sign, pT rounded to 5 GeV, eta in units of 0.5; and the visible mask."""
    x = np.asarray(x, dtype=np.float64)
    typ = np.rint(x[..., 5]).astype(np.int64)
    vis = (x[..., 7] > 0.5) & (typ >= 0) & (typ <= 2)
    sign = np.sign(np.rint(np.nan_to_num(x[..., 4]))).astype(np.int64)
    pt5 = np.floor(np.maximum(np.nan_to_num(x[..., 0]), 0.0) / 5.0 + 0.5).astype(np.int64) * 5
    k = np.floor(np.nan_to_num(x[..., 1]) * 2.0 + 0.5).astype(np.int64)
    return np.stack([typ, sign, pt5, k], -1), vis


def caption_text(t, s, pt5, k):
    return f"{TYPE_NAME[int(t)]}, {SIGN_NAME[int(s)]} charge, pT {int(pt5)} GeV, eta {int(k) / 2:.1f}"


class WarpLand:
    """Caption -> 8-vector. Known captions: the shipped table. Unknown ones: frozen Galactica-6.7B (optional)."""

    def __init__(self, root=ROOT, galactica=None):
        d = os.path.join(root, "land")
        self.captions = json.load(open(os.path.join(d, "captions.json")))
        self.table = np.load(os.path.join(d, "psi_gal6.7b_L24.npy")).astype(np.float32)
        extra = os.path.join(d, "captions_extra.json")     # captions of the shipped examples, embedded by Galactica
        if os.path.exists(extra):
            self.captions += json.load(open(extra))
            self.table = np.concatenate([self.table, np.load(os.path.join(d, "psi_extra.npy")).astype(np.float32)])
        self.index = {c: i for i, c in enumerate(self.captions)}
        p = np.load(os.path.join(d, "gal6.7b_L24_projection.npz"))
        self.mu, self.comps, self.zm, self.zs = p["pooled_mean"], p["components"], p["z_mean"], p["z_std"]
        self.galactica = galactica          # None, or a hub id / local path to load lazily
        self._tok = self._lm = None

    def _embed(self, texts):
        if self.galactica is None:
            raise KeyError(f"{len(texts)} captions are not in the shipped table; pass galactica='{GALACTICA}'")
        from transformers import AutoModel, AutoTokenizer
        if self._lm is None:
            self._tok = AutoTokenizer.from_pretrained(self.galactica)
            self._lm = AutoModel.from_pretrained(self.galactica, dtype=torch.float16).eval().cuda()
            for q in self._lm.parameters():
                q.requires_grad_(False)                       # frozen, never trained
            if self._tok.pad_token is None:                   # same convention as the table build
                pad_id = self._lm.config.pad_token_id if self._lm.config.pad_token_id is not None else 1
                self._tok.pad_token = self._tok.convert_ids_to_tokens(pad_id)
            self._tok.padding_side = "right"
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), 128):
                enc = self._tok(texts[s:s + 128], padding=True, truncation=True, max_length=MAX_LEN,
                                return_tensors="pt").to("cuda")
                h = self._lm(**enc, output_hidden_states=True).hidden_states[LAYER].float()
                m = enc["attention_mask"].unsqueeze(-1).float()
                pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu().numpy()
                out.append(pooled.astype(np.float16).astype(np.float64))   # fp16 pooled, as in the table build
        z = (np.concatenate(out) - self.mu) @ self.comps.T
        return ((z - self.zm) / self.zs).astype(np.float32)

    def object_index(self, x, hide=None):
        """(N,15) caption index into self.table (extended with new captions as needed), -1 = not deposited.
        hide: optional (N,) slot index per event to leave out (a hidden object)."""
        rec, vis = caption_records(x)
        if hide is not None:
            vis = vis & (np.arange(15)[None] != np.asarray(hide)[:, None])
        idx = np.full(vis.shape, -1, dtype=np.int64)
        new = []
        for n, s in zip(*np.nonzero(vis)):
            c = caption_text(*rec[n, s])
            if c not in self.index:
                new.append(c); self.index[c] = len(self.captions) + len(new) - 1
            idx[n, s] = self.index[c]
        if new:
            self.table = np.concatenate([self.table, self._embed(new)])
            self.captions += new
        return idx

    def channels(self, x, hide=None):
        return build_land(x, self.object_index(x, hide), self.table)


class WatershedJEPA:
    def __init__(self, root=ROOT, device=None, galactica=None):
        from safetensors.torch import load_file
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LandJEPA().to(self.device).eval()
        self.model.load_state_dict(load_file(os.path.join(root, "model.safetensors")))
        self.land = WarpLand(root, galactica)

    def inputs(self, x, hide=None):
        x = np.asarray(x, dtype=np.float32).copy()
        if hide is not None:
            x[np.arange(len(x)), np.asarray(hide), 7] = 0.0
        h = build_terrain(x).astype(np.float16).astype(np.float32)   # the training terrains were stored fp16
        land = self.land.channels(x, hide)
        return torch.as_tensor(h, device=self.device), torch.as_tensor(land, device=self.device)

    @torch.no_grad()
    def encode(self, x, window=None, hide=None):
        """Latent (N, 256). window: (N,32,64) bool hidden window, default none hidden (the eval convention)."""
        h, land = self.inputs(x, hide)
        win = torch.zeros_like(h, dtype=torch.bool) if window is None else torch.as_tensor(window, device=self.device)
        return self.model.enc(h, win, land)

    @torch.no_grad()
    def predict_summary(self, x, window):
        """Predicted watershed summary of the hidden window: basin-count probabilities (N,8), persistence (N,8),
        log1p water per basin (N,8)."""
        z = self.encode(x, window)
        logits, pers, water = self.model.heads(z)
        return {"count_probs": logits.softmax(1).cpu().numpy(), "persistence": pers.cpu().numpy(),
                "log1p_water": water.cpu().numpy()}


def exact_summary(x, window):
    """True watershed summary of the window from the FULL event, by the exact hydraulics (CPU)."""
    from hydraulics import flood, labels_at, lifetimes, window_targets
    h = build_terrain(np.asarray(x, dtype=np.float32)).astype(np.float16).astype(np.float64)
    N = h.shape[0]
    labs = np.zeros((N, h.shape[1] * h.shape[2]), np.int64)
    life = np.zeros_like(labs, dtype=np.float32)
    for i in range(N):
        b, d, dp = flood(h[i])
        labs[i] = labels_at(h[i], d, dp, TAU)
        life[i] = lifetimes(b, d, float(h[i].max())).astype(np.float16)
    tg = window_targets(torch.as_tensor(h.reshape(N, -1), dtype=torch.float32), torch.as_tensor(labs),
                        torch.as_tensor(life), torch.as_tensor(np.asarray(window)).reshape(N, -1))
    return {"count": tg["count"].clamp(0, NCLS - 1).numpy(), "persistence": tg["pers"].numpy(),
            "water": tg["water"].numpy()}


def window_mask(r0, c0, n=1):
    """An 8 x 16 window starting at eta row r0, phi column c0 (phi wraps)."""
    rows, cols = np.arange(32), np.arange(64)
    m = (rows[:, None] >= r0) & (rows[:, None] < r0 + WIN_R) & (((cols[None] - c0) % 64) < WIN_C)
    return np.repeat(m[None], n, 0)
