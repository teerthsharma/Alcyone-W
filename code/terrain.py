"""Event -> terrain on the (eta, phi) cylinder: 32 x 64 grid, eta clamped to [-2.5, 2.5] (not periodic),
phi periodic. Each valid object deposits its pT with a Gaussian of sigma = 1 cell, truncated to a 7 x 7
footprint and normalised over the in-range footprint (so the deposit conserves the summed pT exactly).
Terrain h = -log(1 + deposit): energy concentrations are basins (minima of h)."""
import glob
import os

import numpy as np

G_ETA, G_PHI = 32, 64
ETA_MIN, ETA_MAX = -2.5, 2.5
SIGMA = 1.0
R = 3
R1 = os.environ.get("ALCYONE_DATA", "release_data")   # folder of CMS shards (only for load_x)


def build_deposit(x):
    """x (N, 15, 8) raw shard slots -> deposit (N, 32, 64) float64."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    valid = x[..., 7] > 0.5
    pt = np.nan_to_num(x[..., 0])
    pt = np.where(valid & (pt > 0), pt, 0.0)
    eta = np.clip(np.nan_to_num(x[..., 1]), ETA_MIN, ETA_MAX)
    phi = np.mod(np.nan_to_num(x[..., 2]), 2 * np.pi)
    ci = (eta - ETA_MIN) / (ETA_MAX - ETA_MIN) * G_ETA - 0.5      # continuous row of the object
    cj = phi / (2 * np.pi) * G_PHI - 0.5                          # continuous column (phi periodic)
    ri0 = np.rint(ci).astype(np.int64)
    rj0 = np.rint(cj).astype(np.int64)
    di = np.arange(-R, R + 1)
    rows = ri0[..., None, None] + di[None, None, :, None]          # (N,15,7,1)
    cols = rj0[..., None, None] + di[None, None, None, :]          # (N,15,1,7) unwrapped
    d2 = (rows - ci[..., None, None]) ** 2 + (cols - cj[..., None, None]) ** 2
    w = np.exp(-d2 / (2 * SIGMA ** 2))
    inr = (rows >= 0) & (rows < G_ETA)
    w = np.where(inr, w, 0.0)
    w = w / w.sum(axis=(2, 3), keepdims=True)                      # exact pT conservation
    rows_c = np.clip(rows, 0, G_ETA - 1)
    cols_w = np.mod(cols, G_PHI)
    n_idx = np.arange(N, dtype=np.int64)[:, None, None, None]
    flat = n_idx * (G_ETA * G_PHI) + rows_c * G_PHI + cols_w       # broadcast to (N,15,7,7)
    vals = pt[..., None, None] * w
    flat = np.broadcast_to(flat, vals.shape)
    keep = np.broadcast_to(inr, vals.shape) & (vals > 0)
    dep = np.bincount(flat[keep].ravel(), weights=vals[keep].ravel(), minlength=N * G_ETA * G_PHI)
    return dep.reshape(N, G_ETA, G_PHI)


def build_terrain(x):
    """x (N, 15, 8) -> h (N, 32, 64) float32 = -log(1 + deposit)."""
    return (-np.log1p(build_deposit(x))).astype(np.float32)


def shard_files(split):
    return sorted(glob.glob(os.path.join(R1, split, "shard_%s_*.npz" % split)))


def load_x(split):
    """All raw slots of a split, shards concatenated in sorted filename order (same order as prep_*.npz)."""
    xs = [np.load(f)["x"] for f in shard_files(split)]
    return np.concatenate(xs)
