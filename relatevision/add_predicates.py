"""Add NEW relation phrases to the predicate bank (true open vocabulary).

Run through setup_mac.sh, which provides the torch environment it needs:

    ./setup_mac.sh --add-predicates "high-fiving, repairing, kneading dough"

The runtime never runs a language model: it slices rows out of
predicate_bank.npz. A phrase that is not in the bank needs, once:
  * W      its embedding from the model's own distilled text student
           (text_student.pt, ~9M params), averaged over the training templates
  * alpha  its spatial/semantic routing weight from the checkpoint's gate MLP
  * type   spatial or semantic (corpus map if known, else alpha >= 0.5)
New rows get no calibrated threshold (NaN), so they use the global floor.

Before writing, the tool re-encodes phrases already in the bank and refuses
to continue unless they match the shipped embeddings (cos >= 0.999) - the
guard against a wrong tokenizer/student producing a silently corrupt bank.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DIST = REPO / "deploy" / "dist" / "relsgg-vits16plus"
CACHE = REPO / "checkpoints" / "relsgg-vits16plus"
HF_REPO = "maelic/relsgg-vits16plus"


def fetch(name: str) -> Path:
    p = CACHE / name
    if not p.exists():
        import urllib.request
        CACHE.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{name}"
        print(f"[add] downloading {name} ...", flush=True)
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(p)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phrases", help="comma-separated phrases to add")
    args = ap.parse_args()
    phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]

    import torch
    import torch.nn.functional as F
    from transformers import CLIPTokenizerFast
    from relsgg.decompose import type_vector
    from relsgg.text.student import PredicateTextStudent
    from relsgg.vocabulary import TRAIN_TEMPLATES

    bank_path = DIST / "predicate_bank.npz"
    z = np.load(bank_path, allow_pickle=True)
    bank = {k: z[k] for k in z.files}
    names = [str(x) for x in bank["names"]]
    new = [p for p in dict.fromkeys(phrases) if p not in names]
    if not new:
        print("[add] all phrases are already in the bank")
        return

    for f in ("tokenizer.json", "tokenizer_config.json"):
        fetch(f)
    student = PredicateTextStudent.from_checkpoint(str(fetch("text_student.pt")), "cpu")
    student._tokenizer = CLIPTokenizerFast.from_pretrained(str(CACHE))

    def encode(strs):
        with torch.no_grad():
            e = sum(student.encode_texts([t.format(p=s) for s in strs])
                    for t in TRAIN_TEMPLATES)
        return F.normalize(e, dim=-1).float()

    # guard: shipped rows must re-encode to themselves
    probe = names[:24]
    cos = (encode(probe) * torch.from_numpy(bank["W"][:24].astype(np.float32))).sum(-1)
    print(f"[add] text-space check on {len(probe)} shipped phrases: min cos {float(cos.min()):.5f}")
    if float(cos.min()) < 0.999:
        raise SystemExit("[add] the encoder does not reproduce the shipped bank - "
                         "refusing to write a corrupt bank")

    ck = torch.load(str(fetch("model.pth")), map_location="cpu", weights_only=False)
    sd = ck.get("ema_model") or ck.get("model") or ck
    W = encode(new)
    gm = torch.nn.Sequential(torch.nn.Linear(W.shape[1], 128), torch.nn.GELU(),
                             torch.nn.Linear(128, 1))
    with torch.no_grad():
        for i in (0, 2):
            gm[i].weight.copy_(sd[f"vocab_head.gate_mlp.{i}.weight"])
            gm[i].bias.copy_(sd[f"vocab_head.gate_mlp.{i}.bias"])
        alpha = torch.sigmoid(gm(W).squeeze(-1)).numpy().astype(np.float32)
    cmap = json.loads((REPO / "relsgg" / "corpus_type_map.json").read_text())
    is_sp, src = type_vector(new, corpus_map=cmap, alpha=alpha)

    backup = DIST / "predicate_bank.orig.npz"
    if not backup.exists():
        shutil.copy2(bank_path, backup)
    n = len(new)
    nan = np.full(n, np.nan, np.float32)
    add = {"names": np.array(new, dtype=object), "W": W.numpy().astype(bank["W"].dtype),
           "alpha": alpha.astype(bank["alpha"].dtype), "thr": nan, "best_f1": nan,
           "is_spatial": is_sp.astype(bank["is_spatial"].dtype),
           "type_source": np.array(list(src), dtype=object),
           "recall": nan, "gt": np.zeros(n, bank["gt"].dtype)}
    out = dict(bank)
    for k, v in add.items():
        out[k] = np.concatenate([bank[k], v])
    np.savez(bank_path, **out)
    for p_, a_, s_ in zip(new, alpha, is_sp):
        print(f"[add] + {p_!r:32s} {'spatial' if s_ else 'semantic':8s} (alpha {a_:.2f})")
    print(f"[add] bank now has {len(out['names'])} phrases "
          f"(original kept at {backup.name})")


if __name__ == "__main__":
    main()
