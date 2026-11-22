"""Alcyone as a tool-using agent: frozen Galactica-6.7B speaks, the W-JEPA and the exact lemma circuits are its tools.

This is prompting, not training. Galactica-6.7B is a base model (not instruction-tuned), so the loop uses a few-shot
format: the model writes `TOOL: name(arg)`, the runtime runs the tool and appends `RESULT: ...`, until the model
writes `ANSWER: ...`. The rules in ALCYONE_RULES.md are prepended to every prompt.

Tools (all deterministic):
  describe(k)      captions of the visible objects of event k
  watershed(k)     W-JEPA prediction vs exact hydraulics for a fixed hidden window of event k
  charge(k)        Gauss-law lemma: total visible lepton charge, and the charge a hidden lepton must carry if the event is neutral
  calc(expr)       arithmetic (numbers, + - * / ** ( ), sqrt, log, exp, pi)
Usage: python code/agent.py "How many basins does event 3 have in its hidden window?" [--galactica facebook/galactica-6.7b]
"""
import argparse, ast, math, operator, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from alcyone import WatershedJEPA, exact_summary, caption_records, caption_text, window_mask, GALACTICA  # noqa: E402

FEWSHOT = """Question: What is 3 times 14?
TOOL: calc(3*14)
RESULT: 42
ANSWER: 42.

Question: What charge must the hidden lepton of event 0 carry?
TOOL: charge(0)
RESULT: visible lepton charge sum = 1; if the event is neutral the hidden lepton has charge -1 (Gauss-law lemma)
ANSWER: -1, if the event's total lepton charge is zero (that is an assumption, not a measurement).

Question: Who are you?
TOOL: none()
RESULT: no tool needed
ANSWER: I am Alcyone, a W-JEPA built by Teerth Sharma. A frozen Galactica-6.7B builds my land, and I predict the watershed of hidden regions of collision events.
"""

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUN = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}


def calc(expr):
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.Name) and n.id == "pi": return math.pi
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100: raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and len(n.args) == 1:
            return _FUN[n.func.id](ev(n.args[0]))
        raise ValueError("unsupported expression")
    return f"{ev(ast.parse(expr, mode='eval')):.6g}"


# Questions about identity are answered from fixed text, never by the base model, so the answer is always honest.
IDENTITY = [
    (r"\b(which|what)\s+(language\s+)?model\b|\bare you\s+(chat\s?gpt|gpt|claude|llama|gemini|an? (llm|chatbot|language model))"
     r"|\bwhat are you\b|\bbased on\b|\bunder the hood\b|\bhow (big|many parameters)\b",
     "I am Alcyone, a W-JEPA (watershed joint-embedding predictive architecture). The part that learned is a "
     "5.9M-parameter watershed JEPA, trained by Teerth Sharma on CMS Open Data collisions. The text you are reading "
     "is written by Galactica-6.7B by Meta AI, used frozen: it was never fine-tuned or instruction-tuned. I am not "
     "ChatGPT, Claude, Llama or any chat model. In total about 6.66B parameters, about 5.9M of them trained."),
    (r"\bwho\s+is\s+teerth\b|\babout\s+teerth\b|\bteerth\s+sharma\b",
     "Teerth Sharma built me. Teerth is an independent researcher who builds at teerth.dev and writes at teerth.blog, "
     "and designed Alcyone from their own proved lemmas (Lean 4 proofs and Haskell mirrors): compute what the lemmas "
     "determine, learn only what they leave open. I am Teerth's first model of this design."),
    (r"\bwho\s+(built|made|created|trained|designed)\s+you\b|\byour\s+(creator|author|maker)\b",
     "Teerth Sharma built me: Teerth designed the W-JEPA and trained its 5.9M learned parameters. The frozen "
     "Galactica-6.7B that writes this text is by Meta AI and was not trained by anyone for Alcyone."),
    (r"\bwho\s+are\s+you\b|\byour\s+name\b|\bintroduce\s+yourself\b",
     "I am Alcyone, a W-JEPA built by Teerth Sharma. A frozen Galactica-6.7B builds my land and writes my text, "
     "and I predict the watershed of hidden regions of collision events."),
]


def identity_answer(question):
    q = question.lower()
    for pat, ans in IDENTITY:
        if re.search(pat, q):
            return ans
    return None


class Tools:
    def __init__(self, jepa, events):
        self.jepa, self.x = jepa, events

    def _k(self, a):
        k = int(a)
        if not 0 <= k < len(self.x): raise ValueError(f"event index must be 0..{len(self.x) - 1}")
        return self.x[k:k + 1]

    def describe(self, a):
        x = self._k(a); rec, vis = caption_records(x)
        caps = [caption_text(*rec[0, s]) for s in range(15) if vis[0, s]]
        return "; ".join(caps) if caps else "no visible objects"

    def watershed(self, a):
        x = self._k(a); win = window_mask(12, 24)
        p = self.jepa.predict_summary(x, win)["count_probs"][0]
        t = exact_summary(x, win)["count"][0]
        return (f"hidden window eta rows 12-19, phi columns 24-39: W-JEPA predicts {int(p.argmax())} basins "
                f"(p={p.max():.2f}); exact hydraulics: {int(t)} basins")

    def charge(self, a):
        x = self._k(a)[0]
        lep = (x[:, 7] > 0.5) & np.isin(x[:, 5], (0.0, 1.0))
        s = int(round(float(x[lep, 4].sum())))
        return (f"visible lepton charge sum = {s}; if the event is neutral the hidden lepton has charge {-s} "
                f"(Gauss-law lemma)") if s != 0 else "visible lepton charge sum = 0; the lemma leaves the hidden charge undetermined"

    def calc(self, a):
        return calc(a)

    def run(self, name, arg):
        if name not in ("describe", "watershed", "charge", "calc"):
            return f"unknown tool {name}"
        try:
            return getattr(self, name)(arg.strip())
        except Exception as e:  # tool errors go back to the model as text
            return f"error: {e}"


class Alcyone:
    def __init__(self, galactica=GALACTICA, root=ROOT, events=None, max_new=96, max_steps=4):
        self.jepa = WatershedJEPA(root, galactica=galactica)
        if events is None:
            events = np.load(os.path.join(root, "examples", "cms_test_sample.npz"))["x"]
        self.tools = Tools(self.jepa, events)
        self.rules = open(os.path.join(root, "ALCYONE_RULES.md")).read()
        self.max_new, self.max_steps = max_new, max_steps
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(galactica)
        self.lm = AutoModelForCausalLM.from_pretrained(galactica, dtype=torch.float16).eval().cuda()
        for q in self.lm.parameters():
            q.requires_grad_(False)                                   # frozen, never trained
        if self.tok.pad_token is None:
            pid = self.lm.config.pad_token_id if self.lm.config.pad_token_id is not None else 1
            self.tok.pad_token = self.tok.convert_ids_to_tokens(pid)
        self.tok.padding_side = "right"
        L = self.jepa.land                                            # one copy of the frozen weights:
        L._tok, L._lm = self.tok, self.lm.model                       # the land embeds with the same decoder

    def _generate(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            out = self.lm.generate(ids, max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def ask(self, question, verbose=False):
        """Runtime-guarded loop (the base model is not trusted to follow the format):
        1. the model must open with a tool choice: the runtime prefills 'TOOL:' and parses name(arg); none() is allowed;
        2. after each RESULT the runtime prefills 'ANSWER:' (one tool call per question, then the answer);
        3. rule 5 check: every number in the answer must appear in a RESULT or the question, else it is flagged."""
        fixed = identity_answer(question)
        if fixed is not None:          # identity and model questions: fixed honest text, never generated
            return {"tool": "identity()", "result": "fixed text", "answer": fixed,
                    "trace": ["TOOL: identity()", "RESULT: fixed text (rules 1 and 7)", f"ANSWER: {fixed}"]}
        tools = ("Tools: describe(k), watershed(k), charge(k) for event k of the loaded sample "
                 f"(0..{len(self.tools.x) - 1}), calc(expr), none().\n\n")
        prompt = self.rules + "\n" + tools + FEWSHOT + f"\nQuestion: {question}\nTOOL:"
        text = self._generate(prompt).split("\n")[0]
        m = re.match(r"\s*(\w+)\((.*)\)", text)
        name, arg = (m.group(1), m.group(2)) if m else ("none", "")
        res = "no tool needed" if name == "none" else self.tools.run(name, arg)
        step = f" {name}({arg})\nRESULT: {res}\nANSWER:"
        prompt += step
        ans = self._generate(prompt).split("\n")[0].strip()
        nums = lambda t: set(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)(?!\.\d)", t))
        unsupported = sorted(nums(ans) - nums(res) - nums(question) - nums(self.rules) - nums(FEWSHOT))
        out = {"tool": f"{name}({arg})", "result": res, "answer": ans,
               "trace": [f"TOOL: {name}({arg})", f"RESULT: {res}", f"ANSWER: {ans}"]}
        if unsupported:
            out["flag"] = f"rule 5: numbers not from a tool: {unsupported}"
            out["trace"].append("FLAG: " + out["flag"])
        return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("question"); ap.add_argument("--galactica", default=GALACTICA)
    a = ap.parse_args()
    r = Alcyone(a.galactica).ask(a.question)
    print("\n".join(r["trace"]))
