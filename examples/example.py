"""Run Alcyone (W-JEPA + Galactica-6.7B warp land) on 512 real CMS Open Data test events (held-out file).
Prints the predicted vs exact basin count of a hidden 8 x 16 window, and the latent shape.
python examples/example.py            (shipped land table only; no Galactica download needed for these events)
python examples/example.py --galactica facebook/galactica-6.7b   (embed any caption not in the table)"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from alcyone import WatershedJEPA, exact_summary  # noqa: E402
from model import sample_windows  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--galactica", default=None); a = ap.parse_args()
x = np.load(os.path.join(ROOT, "examples", "cms_test_sample.npz"))["x"]
m = WatershedJEPA(ROOT, galactica=a.galactica)
win = sample_windows(len(x), "cpu", gen=torch.Generator().manual_seed(0)).numpy()
pred = m.predict_summary(x, win)["count_probs"].argmax(1)
true = exact_summary(x, win)["count"]
maj = np.bincount(true).argmax()
print("latent", tuple(m.encode(x[:4]).shape))
print(f"basin count of the hidden window: Alcyone {np.mean(pred == true):.3f}, majority guess {np.mean(true == maj):.3f}, n={len(x)}")
