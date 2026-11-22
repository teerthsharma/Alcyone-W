"""Land-channel builder: per-object 8-vectors deposited on the cylinder with terrain.py's kernel.

Same geometry, footprint (7 x 7, R = 3), Gaussian (SIGMA = 1 cell), eta clamp, phi wrap and per-object
normalisation over the in-range footprint as terrain.build_deposit, so every channel conserves the sum of
its per-object values exactly (deposit.sum(cells) == sum over deposited objects of psi).
Test: with psi = 1 and every valid non-MET object deposited, the channel equals terrain.build_deposit on pT=1.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from terrain import G_ETA, G_PHI, ETA_MIN, ETA_MAX, SIGMA, R  # noqa: E402

N_LAND = 8


def build_land(x, obj_cap, table):
    """x (N,15,8) raw slots (only eta, phi used here), obj_cap (N,15) int (-1 = not deposited),
    table (U,C) float per-caption vectors. Returns (N, C, 32, 64) float32."""
    x = np.asarray(x, dtype=np.float64)
    N, S = x.shape[0], x.shape[1]
    table = np.asarray(table, dtype=np.float64)
    C = table.shape[1]
    dep = obj_cap >= 0
    psi = table[np.clip(obj_cap, 0, None)] * dep[..., None]           # (N,15,C), zero where not deposited
    eta = np.clip(np.nan_to_num(x[..., 1]), ETA_MIN, ETA_MAX)
    phi = np.mod(np.nan_to_num(x[..., 2]), 2 * np.pi)
    ci = (eta - ETA_MIN) / (ETA_MAX - ETA_MIN) * G_ETA - 0.5
    cj = phi / (2 * np.pi) * G_PHI - 0.5
    ri0 = np.rint(ci).astype(np.int64)
    rj0 = np.rint(cj).astype(np.int64)
    di = np.arange(-R, R + 1)
    rows = ri0[..., None, None] + di[None, None, :, None]
    cols = rj0[..., None, None] + di[None, None, None, :]
    d2 = (rows - ci[..., None, None]) ** 2 + (cols - cj[..., None, None]) ** 2
    w = np.exp(-d2 / (2 * SIGMA ** 2))
    inr = (rows >= 0) & (rows < G_ETA)
    w = np.where(inr, w, 0.0)
    w = w / w.sum(axis=(2, 3), keepdims=True)                          # sums to 1 per object
    rows_c = np.clip(rows, 0, G_ETA - 1)
    cols_w = np.mod(cols, G_PHI)
    cell = (rows_c * G_PHI + cols_w).reshape(N, S, -1)                 # (N,15,49)
    wf = w.reshape(N, S, -1)
    n_idx = np.arange(N, dtype=np.int64)[:, None, None, None]
    c_idx = np.arange(C, dtype=np.int64)[None, None, :, None]
    flat = (n_idx * C + c_idx) * (G_ETA * G_PHI) + cell[:, :, None, :]  # (N,15,C,49)
    vals = psi[:, :, :, None] * wf[:, :, None, :]                      # (N,15,C,49)
    out = np.bincount(flat.ravel(), weights=vals.ravel(), minlength=N * C * G_ETA * G_PHI)
    return out.reshape(N, C, G_ETA, G_PHI).astype(np.float32)
