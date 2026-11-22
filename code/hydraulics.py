"""Exact union-find merge tree (sublevel sweep) on the periodic cylinder, plus per-window targets.

Lemma semantics (lean/WatershedLemmas): W1 beta0 changes by 0 or -1 per edge; W2 elder rule: on a
merge the survivor keeps the lower minimum (ties: lower cell index), the younger dies at the merge
height; W3 water conservation under merging; W4 merges = (#basins - 1).
Basin label of a cell = index of the elder's birth cell (its minimum)."""
import numpy as np
import torch

G_ETA, G_PHI = 32, 64
_NB = {}


def neighbours(g0, g1):
    key = (g0, g1)
    if key not in _NB:
        nb = np.full((g0 * g1, 8), -1, dtype=np.int64)
        offs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for i in range(g0):
            for j in range(g1):
                for k, (di, dj) in enumerate(offs):
                    ii = i + di
                    if 0 <= ii < g0:
                        nb[i * g1 + j, k] = ii * g1 + (j + dj) % g1      # phi periodic, eta not
        _NB[key] = nb
    return _NB[key]


def flood(h2d):
    """Sublevel sweep (ascending h) with union-find and the elder rule.
    Returns birth (N,) = h of each cell (its own birth), death (N,) = merge height at which the cell's
    component died into an elder (+inf for the global survivor), dpar (N,) = the elder root it died into
    (self for survivors). The merge tree is (birth, death, dpar)."""
    h = np.asarray(h2d, dtype=np.float64)
    g0, g1 = h.shape
    hf = h.reshape(-1)
    N = hf.size
    nb = neighbours(g0, g1)
    order = np.lexsort((np.arange(N), hf))
    added = np.zeros(N, dtype=bool)
    par = np.arange(N)
    dpar = np.arange(N)
    death = np.full(N, np.inf)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for p in order:
        p = int(p)
        added[p] = True
        hp = hf[p]
        for q in nb[p]:
            if q < 0:
                continue
            q = int(q)
            if not added[q]:
                continue
            rp, rq = find(p), find(q)
            if rp == rq:
                continue
            # elder = lower birth height, ties by lower index (the survivor keeps the min label)
            if (hf[rp], rp) < (hf[rq], rq):
                elder, young = rp, rq
            else:
                elder, young = rq, rp
            death[young] = hp
            dpar[young] = elder
            par[young] = elder
    return hf.copy(), death, dpar


def labels_at(h2d, death, dpar, tau):
    """Basin label per cell = the elder root of the sublevel component {h <= tau} containing the cell.
    -1 for cells with h > tau (not in the sublevel set). Climbs the merge tree: a cell follows dpar while its
    component has already merged by level tau."""
    hf = np.asarray(h2d, dtype=np.float64).reshape(-1)
    x = np.arange(hf.size)
    active = hf <= tau
    x = np.where(active, x, 0)
    while True:
        dead = death[x] <= tau
        if not dead.any():
            break
        x = np.where(dead, dpar[x], x)
    return np.where(active, x, -1)


def alive_count(birth, death, t):
    """Number of sublevel-set components {h <= t} from the merge tree: components with birth <= t < death."""
    b = np.asarray(birth)
    d = np.asarray(death)
    return int(np.sum((b <= t) & (t < d)))


def lifetimes(birth, death, h_max):
    """Per-cell lifetime (death - birth); the global survivor's death is taken as the terrain top h_max."""
    d = np.where(np.isfinite(death), death, h_max)
    return d - birth


def water_check(lab, deposit_flat, mask_flat, drop_label=None):
    """W3 check: per-basin water ledger over the window (active cells only) must sum to the window's total
    active deposit. drop_label simulates a wrong assignment (one basin's water is lost from the ledger)."""
    ledger = {}
    m_arr = np.asarray(mask_flat, dtype=bool)
    for r, w, m in zip(lab, deposit_flat, m_arr):
        if m and r >= 0:
            ledger[int(r)] = ledger.get(int(r), 0.0) + float(w)
    if drop_label is not None:
        ledger.pop(int(drop_label), None)
    act = m_arr & (np.asarray(lab) >= 0)
    total = float(np.sum(np.asarray(deposit_flat)[act]))
    return bool(np.isclose(sum(ledger.values()), total, rtol=1e-9, atol=1e-9))


def window_targets(h, lab, life, win, k=8):
    """Torch, batched. h (B,N) terrain, lab (B,N) basin labels (-1 = outside the sublevel set), life (B,N)
    lifetime of each cell's basin root, win (B,N) bool window mask. Returns dict: count (B,) basins touching the window, pers (B,k) top-k
    lifetimes of those basins (desc, zero-padded), water (B,k) top-k per-basin window water (desc,
    zero-padded), water_total (B,) = window deposit, ledger_total (B,) = sum of per-basin water."""
    B, N = lab.shape
    dep = torch.expm1(-h.to(torch.float64)).clamp_min(0.0)
    act = win & (lab >= 0)
    dep_w = dep * act.to(dep.dtype)
    idx = torch.where(act, lab, torch.full_like(lab, N))
    touched = torch.zeros(B, N + 1, dtype=torch.bool, device=lab.device)
    touched.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
    touched = touched[:, :N]
    count = touched.sum(1)
    pers_all = torch.where(touched, life.to(torch.float64), torch.full_like(dep, -1e9))
    pers = pers_all.topk(k, dim=1).values.clamp_min(0.0)
    water_root = torch.zeros(B, N + 1, dtype=torch.float64, device=lab.device)
    water_root.scatter_add_(1, idx, dep_w)
    water_root = water_root[:, :N]
    water = water_root.topk(k, dim=1).values
    return {"count": count, "pers": pers, "water": water,
            "water_total": dep_w.sum(1), "ledger_total": water_root.sum(1)}


def window_targets_np(h_flat, lab, life, win_flat):
    """Single-window reference (used by tests and eval)."""
    out = window_targets(torch.as_tensor(np.asarray(h_flat)[None], dtype=torch.float64),
                         torch.as_tensor(np.asarray(lab)[None]),
                         torch.as_tensor(np.asarray(life)[None], dtype=torch.float64),
                         torch.as_tensor(np.asarray(win_flat)[None], dtype=torch.bool))
    return {"count": int(out["count"][0]), "pers": out["pers"][0].numpy(),
            "water": out["water"][0].numpy(), "water_total": float(out["water_total"][0]),
            "ledger_total": float(out["ledger_total"][0])}
