<div align="center">

# 🌊 Alcyone

### W-JEPA: a joint-embedding predictive architecture that predicts the *watershed* of what it can't see

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-teerthsharma%2FAlcyone--W-yellow)](https://huggingface.co/teerthsharma/Alcyone-W)
[![GitHub](https://img.shields.io/badge/GitHub-teerthsharma%2FAlcyone--W-181717?logo=github)](https://github.com/teerthsharma/Alcyone-W)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
![trained](https://img.shields.io/badge/trained-5.9M-38bdf8)
![frozen](https://img.shields.io/badge/frozen-Galactica--6.7B-a78bfa)
![data](https://img.shields.io/badge/data-CMS%20Open%20Data-f472b6)

### ✨ First of its kind: to our knowledge, the first JEPA whose prediction target is *topology*, a watershed computed exactly from proved lemmas, instead of a learned embedding.

**Built by [Teerth Sharma](https://teerth.dev)** · from proved lemmas (Lean 4 + Haskell) · first model of a new design

<img src="assets/pipeline.svg" alt="Alcyone pipeline: event, land, hidden window, watershed" width="100%"/>

</div>

---

## The idea in one breath

Every JEPA we could find (I-JEPA, V-JEPA, Graph-JEPA, Point-JEPA and the rest) hides part of the input and predicts a learned embedding of it. **Alcyone is the first we know of that predicts its topology instead.** The claim rests on a literature search in November 2026. If you know earlier work, open an issue and we'll credit it.

A particle collision becomes a landscape, and a window of that landscape is hidden. From the rest, Alcyone predicts what rain would do inside the window:
- how many separate lakes (basins) form;
- how long each survives as the water rises (persistence, by the elder rule);
- how much energy each one holds.

The answers aren't another network's guess. They're **computed exactly** by a union-find flood whose rules are proved in Lean, so there's no EMA teacher and no collapse.

> *Compute what the lemmas determine. Learn only what they leave open.*

## What's inside

| | | |
|---|---|---|
| 🌊 **W-JEPA core** | 5,893,976 params, **trained** | conv encoder on the periodic (η, φ) cylinder + watershed predictor |
| 📚 **Galactica-6.7B** | 6,657,359,872 params, **frozen** | captions each particle in plain text and turns it into 8 land channels, *never trained*; loaded from [`facebook/galactica-6.7b`](https://huggingface.co/facebook/galactica-6.7b), not redistributed |
| ⚖️ **Exact hydraulics** | 0 params | union-find flood, elder rule, energy conserved under basin merges: the lemmas, as code |
| 🧰 **Agent** | rules + 4 tools | Galactica speaks, guarded so every number comes from a tool |

**≈ 6.66B parameters total, ≈ 5.9M trained.** Never "6.6B trained".

## How a collision becomes a landscape

1. **Terrain.** Every muon, electron and jet drops its momentum on a 32 × 64 grid wrapped around the beam (η × φ), using a Gaussian kernel that conserves energy *exactly*. Height = −log(1 + deposit), so energy pools into valleys.
2. **Warp land.** Each particle gets a caption, such as `muon, negative charge, pT 40 GeV, eta 1.0`. Frozen Galactica-6.7B reads it (layer 24, mean-pooled), and a fixed PCA turns that into 8 numbers. Those are painted onto the same grid as extra land. The language model shapes the land before the flood.
3. **Hide.** A random 8 × 16 window goes dark. The encoder sees only the rest.
4. **Predict.** The encoder predicts the hidden window's basin count, its top-8 persistences and the energy per basin, matched against the exact flood of the full event.

## Results: the whole truth

**Its own task.** Alcyone gets the hidden window's basin count right:
- **0.945** on 512 real CMS events from a file it never trained on, against 0.572 for always guessing the most common count (`examples/example.py`);
- **0.936** on 4,096 fixed validation windows.

**Downstream physics** (frozen encoder, small MLP probe, 3 probe seeds, 200k held-out events):

| task | **Alcyone** | same W-JEPA, no Galactica land | plain kinematics MLP |
|---|---|---|---|
| jet multiplicity (MSE ↓) | **0.0103** | 0.0796 | 0.0016 |
| dimuon mass, leading muon hidden (MSE ↓) | 25,211 | 25,629 | 24,640 |
| hidden-lepton charge (acc ↑) | 0.503 | 0.502 | 0.500 |

- 🟢 **The language model's text really shapes the land.** On jet multiplicity, the Galactica land cuts error **85–87%** against a flat land, and beats each control on all 3 seeds:
  - shuffled captions: 84–87% lower error;
  - Galactica-125m: 18–23% lower;
  - a no-language-model code: 40–51% lower.
- 🔴 **But it's one task.** The pre-registered bars asked for two of three, so they failed. Alcyone doesn't claim the frozen model helps in general.
- 🔴 **A plain kinematics MLP beats it** on both answerable tasks.
- ⚪ **Hidden-lepton charge is unanswerable as built.** Charge isn't in the terrain, so everything sits at chance. A charge-aware W-JEPA is training as this is written.

Every bar was written down before training. Every failure is reported. Every number traces to a results file.

## Quick start

```bash
git clone https://github.com/teerthsharma/Alcyone-W && cd Alcyone-W
hf download teerthsharma/Alcyone-W --local-dir .        # weights, land table, 512-event sample
pip install -r requirements.txt
python examples/example.py                               # basin count on real held-out collisions
```

```python
import sys; sys.path.insert(0, "code")
from alcyone import WatershedJEPA, exact_summary, window_mask

m   = WatershedJEPA(".", galactica="facebook/galactica-6.7b")  # Galactica only for captions not in the table
z   = m.encode(x)                                  # x: (N, 15, 8) [pT, eta, phi, mass, charge, type, btag, valid]
win = window_mask(12, 24, len(x))                  # hide eta rows 12-19, phi columns 24-39
pred = m.predict_summary(x, win)                   # basin-count probabilities, persistence, water
true = exact_summary(x, win)                       # the exact flood, for comparison
```

## Talk to it

```bash
python code/agent.py "What charge must the hidden lepton of event 5 carry?"
```
```
TOOL: charge(5)
RESULT: visible lepton charge sum = -1; if the event is neutral the hidden lepton has charge 1 (Gauss-law lemma)
ANSWER: 1, if the event's total lepton charge is zero (that is an assumption, not a measurement).
```

Alcyone follows shared **rules** ([`ALCYONE_RULES.md`](ALCYONE_RULES.md)): who it is, exact before learned, every number from a tool. It has **four tools**:

| tool | what it does |
|---|---|
| `watershed(k)` | the W-JEPA's prediction against the exact flood |
| `charge(k)` | the Gauss-law lemma |
| `describe(k)` | lists an event's particles |
| `calc(expr)` | arithmetic |

Galactica is a raw base model, so the runtime doesn't trust it:
- it forces a tool choice first;
- it fills in the answer slot only after the tool result comes back;
- it flags any number in the answer that no tool returned.

Ask it who it is, which model it is, or who built it, and it answers from **fixed text, honestly**: a 5.9M-parameter W-JEPA trained by Teerth Sharma, with frozen Galactica-6.7B writing the words. It's not a chat model, and it says so.

## The lemmas behind it

The rules Alcyone computes exactly are proved, not assumed:
- **The proofs:** Lean 4, 1,818 built declarations with no `sorry`, counted from build manifests.
- **The mirrors:** executable Haskell versions with QuickCheck property tests.
- **The hydraulics:** the watershed rules live in `WatershedLemmas`: the beta0 step, the elder rule, water conserved under spills, n-1 merges and graph Poincaré-Hopf.

Find them in [`lemmas/`](https://huggingface.co/teerthsharma/Alcyone-W/tree/main/lemmas) on Hugging Face.

## Honest limits

- **Narrow:** one detector format (CMS NanoAOD objects) and one task family.
- **Training hid one lepton:** training never captioned the hidden lepton of one probe task, but this release deposits every visible object, a mild shift from training.
- **Window edges leak:** a particle near the edge of the window is partly visible from outside it, so the training task is partly readable from the context.
- **Answers can still go wrong:** the agent picks tools reliably and reports their results correctly, but its free-text answers can add copied phrases or miscount.

## Provenance

- **Data:** CMS Open Data (CERN Open Data portal), NanoAOD.
  - 4M events with at least one lepton, split by source file: 2.24M train, 1.06M val, 693k test.
  - `examples/cms_test_sample.npz` holds 512 test events, with run, lumi and event numbers.
- **Checkpoint:** arm L1, seed 1 of a pre-registered 3-seed run, 38,155 steps. The seed was picked by a rule fixed beforehand: the lowest final training-task loss.
- **Freeze checks, every run:**
  - Galactica's weight sha256 is identical before and after training;
  - every Galactica parameter has requires_grad False;
  - 1,000 captions re-embed bit-identically;
  - the land table's hash matches its provenance.

  `MANIFEST.json` lists a sha256 for every released file.
- **Galactica:** Taylor et al., *Galactica: A Large Language Model for Science* (2022), Meta AI, CC BY-NC 4.0.

## License

**CC BY-NC 4.0** (non-commercial). The land table and projection are derived from Galactica-6.7B outputs, and Galactica is CC BY-NC 4.0. Galactica's weights are not included.

<div align="center"><sub>Alcyone, the brightest star of the Pleiades. Built by <a href="https://teerth.dev">Teerth Sharma</a> · <a href="https://teerth.blog">teerth.blog</a></sub></div>
