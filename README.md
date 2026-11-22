<div align="center">

# 🌊 Alcyone

### A new kind of JEPA: it predicts *topology*

**W-JEPA (watershed JEPA).** Hide part of a real particle-collision event, then predict the shape of what is hidden: how many basins it holds, how long they persist, and how much energy each one carries. The target comes from proved math, not from a second network.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Weights on Hugging Face](https://img.shields.io/badge/🤗%20weights-Alcyone--W-yellow)](https://huggingface.co/teerthsharma/Alcyone-W)
[![Lean 4](https://img.shields.io/badge/Lean%204-1%2C818%20declarations%2C%20no%20sorry-5b3cc4)](https://huggingface.co/teerthsharma/Alcyone-W/tree/main/lemmas)
[![Haskell](https://img.shields.io/badge/Haskell-QuickCheck%20mirrors-5e5086)](https://huggingface.co/teerthsharma/Alcyone-W/tree/main/lemmas)
[![Data](https://img.shields.io/badge/data-CMS%20Open%20Data%2C%204M%20events-0b7285)](https://opendata.cern.ch)

Built by **Teerth Sharma** · [teerth.dev](https://teerth.dev) · [teerth.blog](https://teerth.blog)

<img src="assets/pipeline.svg" alt="Alcyone pipeline: event, land, hidden window, watershed" width="100%"/>

</div>

---

## ✨ Highlights

- 🧭 **A new target for JEPAs.** I-JEPA, V-JEPA, Graph-JEPA and Point-JEPA all predict a *learned embedding* of the hidden part. Alcyone predicts its **topology**: basin count, H0 persistence (elder rule) and energy per basin. To our knowledge it is the first JEPA to do this (literature search, November 2026; earlier work, if found, will be credited).
- 🔒 **No collapse by construction.** The target is an exact union-find flood, so the model needs **no EMA teacher, no stop-gradient tricks and no collapse regularisers**. There is nothing for it to collapse onto.
- 🎯 **94.5% on the hidden watershed**, on events from a held-out source file, against **57.2%** for the majority guess. It scores 93.6% on 4,096 fixed validation windows.
- 🧪 **Trained on real physics.** 4 million proton-proton collisions from **CMS Open Data**, split by source file (2.24M train / 1.06M val / 693k test).
- 🧠 **A frozen 6.7B scientific LLM shapes the terrain.** Galactica-6.7B reads one caption per particle, and its embedding becomes 8 extra land channels. That land **cuts jet-multiplicity error by 85–87%** relative to a flat land, and it wins on **all three training seeds** against every control:
  - shuffled captions: 84–87% lower error;
  - a smaller Galactica-125m: 18–23% lower;
  - a parameter-free caption code: 40–51% lower.
- 📐 **Proved before it was coded.** The flood rules (the beta0 step, the elder rule, conservation under spill, n−1 merges, graph Poincaré–Hopf) are **proved in Lean 4**: 1,818 built declarations with **zero `sorry`**, plus executable **Haskell mirrors** checked with QuickCheck.
- 🪶 **Tiny trained core.** Only **5.9M trained parameters** do the learning. Everything the lemmas determine is computed exactly, so the network learns only what the math leaves open.
- 🔍 **Freeze checks on every run.** Galactica's weight hashes match before and after training, it has no gradients, and 1,000 captions re-embed bit-identically.
- 🧾 **Pre-registered science.** Every bar was written down before training, and every outcome is published, the misses included.

## 🏗️ Architecture

> **Compute what the lemmas determine, learn only what they leave open.**

| component | parameters | role |
|---|---|---|
| W-JEPA encoder + predictor | 5,893,976 | **trained**: predicts the hidden watershed |
| Galactica-6.7B | 6,657,359,872 | **frozen** land builder; loaded from `facebook/galactica-6.7b`, not redistributed |
| exact hydraulics (union-find flood, elder rule, conservation) | 0 | the proved lemmas, implemented as code |

About **6.66B parameters in total**, of which **5.9M are trained**.

1. **Terrain.** Every muon, electron and jet deposits its transverse momentum on a 32 × 64 grid over the (η, φ) cylinder. The Gaussian kernel conserves the total exactly. Height is h = −log(1 + deposit), so energy pools into basins.
2. **Land.** Each object gets a caption such as `muon, negative charge, pT 40 GeV, eta 1.0`. Galactica-6.7B embeds it (layer 24, mean-pooled), a PCA reduces it to 8 dimensions, and the result is painted onto the terrain as 8 more channels.
3. **Mask.** A random 8 × 16 window is hidden. A convolutional encoder on the cylinder sees the rest.
4. **Predict the shape.** The predictor outputs the window's basin count, its top-8 persistences and the energy per basin. These are scored against the exact flood of the full event.

## 🚀 Quickstart

```bash
git clone https://github.com/teerthsharma/Alcyone-W && cd Alcyone-W
hf download teerthsharma/Alcyone-W --local-dir .
pip install -r requirements.txt
python examples/example.py
```

The example runs on 512 real held-out CMS events that ship with the weights. It needs no Galactica download for those events.

```python
import sys; sys.path.insert(0, "code")
from alcyone import WatershedJEPA, exact_summary, window_mask

m = WatershedJEPA(".", galactica="facebook/galactica-6.7b")
z = m.encode(x)                       # x: (N, 15, 8) [pT, eta, phi, mass, charge, type, btag, valid]
win = window_mask(12, 24, len(x))
pred = m.predict_summary(x, win)      # basin-count probabilities, persistence, energy per basin
true = exact_summary(x, win)          # exact flood, the ground truth
```

## 🤖 Talk to it: the agent

`code/agent.py` puts Alcyone behind the frozen Galactica-6.7B, with four deterministic tools:

| tool | what it does |
|---|---|
| `watershed(k)` | W-JEPA prediction next to the exact flood |
| `charge(k)` | Gauss-law readout |
| `describe(k)` | object list of event k |
| `calc(expr)` | arithmetic |

The runtime enforces the protocol: the agent picks a tool first, fills the answer only after the tool returns, and flags any number that no tool produced. Ask it who it is and it answers from fixed text ([`ALCYONE_RULES.md`](ALCYONE_RULES.md)): a 5.9M-parameter W-JEPA by Teerth Sharma, speaking through a frozen Galactica-6.7B. It is not a chat model.

## 📊 Full results

| task (frozen encoder, MLP probe, 3 seeds, 200k held-out events) | Alcyone | same W-JEPA, flat land | kinematics MLP |
|---|---|---|---|
| jet multiplicity (MSE) | **0.0103** | 0.0796 | 0.0016 |
| dimuon mass, leading muon hidden (MSE) | 25,211 | 25,629 | 24,640 |
| hidden-lepton charge (accuracy) | 0.503 | 0.502 | 0.500 |

The language-model land is a clear win for jet multiplicity. The pre-registered land bars (WL1–WL5) asked for a win on two of three tasks, and this release gets one. A plain kinematics MLP still leads on the two answerable tasks. Charge cannot be answered yet because the terrain carries no charge; a charge-aware variant is being built.

## 🛣️ What's next

- 🔋 **Charge-aware terrain**, so the hidden-lepton task becomes answerable.
- 🧱 **Edge-clean windows**: deposits near the window edge are partly visible from outside, and closing that leak makes the task harder and more honest.
- 🌍 **More detectors and task families** beyond CMS NanoAOD objects.
- 🗣️ **Better free-text answers** from the agent; its tool choices and tool outputs are already reliable.

## 📦 Data and provenance

- **Data:** CMS Open Data (CERN Open Data portal), NanoAOD. 4M events with at least one lepton, split by source file. 512 test events ship with the weights.
- **Checkpoint:** training seed 1 of three, 38,155 steps, chosen by a rule fixed in advance (lowest final masked-summary loss).
- **Galactica:** Taylor et al., *Galactica: A Large Language Model for Science*, Meta AI, 2022.

## 📜 License

CC BY-NC 4.0. The land table and projection derive from Galactica-6.7B outputs, and Galactica is licensed CC BY-NC 4.0. Galactica's weights are not included.

<div align="center">

⭐ **If a JEPA that predicts topology sounds fun to you, star the repo and say hi!** ⭐

</div>
