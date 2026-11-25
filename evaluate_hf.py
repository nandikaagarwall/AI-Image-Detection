"""
Evaluate pretrained open-source detectors (hf_detectors.py) AND your CIFAKE models on
  * AI-generated images  (default: evaluation/images/pollinations)  -> label 1
  * real photographs     (default: evaluation/images/real, see fetch_real_photos.py) -> label 0
under several perturbations, plus an optional CIFAKE in-domain check.

  python evaluate_hf.py --cifake "C:/Users/you/Downloads/nandika_project.zip"
  python evaluate_hf.py --models commfor,haywoodsloan                  # subset
  python evaluate_hf.py --ai-dir evaluation/images/gemini --real-dir my_photos

Outputs (one folder per model):
  evaluation/open_source/<model>/per_image.csv   P(AI) per image x condition
  evaluation/open_source/<model>/report.md       that model's full evaluation
  evaluation/open_source/<model>/summary.json
  evaluation/open_source/leaderboard.md / .csv   all models side by side
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import hf_detectors
from evaluate import CONDITIONS, OUT, auc, load_cifake_sample

ROOT = Path(__file__).parent
HF_OUT = OUT / "open_source"
HF_CONDITIONS = {k: v[0] for k, v in CONDITIONS.items() if "TTA" not in k}  # name -> transform
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def load_dir(d: Path, label: int):
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT) if d.exists() else []
    return [(f"{d.name}/{f.name}", label, Image.open(f).convert("RGB")) for f in files]


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= 0.5
    out = {"acc": float((pred == y).mean())}
    if (y == 1).any():
        out["tpr"] = float(pred[y == 1].mean())  # AI images caught
    if (y == 0).any():
        out["fpr"] = float(pred[y == 0].mean())  # real photos wrongly flagged
    out["auc"] = float(auc(y, p))
    return out


def evaluate_model(det, items, cifake):
    rows, t0, n = [], time.time(), 0
    for cond, fn in HF_CONDITIONS.items():
        p = det.predict([fn(im) for _, _, im in items])
        n += len(items)
        rows += [{"image": name, "label": "AI" if y else "real", "condition": cond, "P_AI": float(pi)}
                 for (name, y, _), pi in zip(items, p)]
    per_image = pd.DataFrame(rows)

    s = {"model": det.key, "repo": det.repo, "description": det.desc}
    for cond in HF_CONDITIONS:
        sub = per_image[per_image.condition == cond]
        for k, v in metrics((sub.label == "AI").astype(int), sub.P_AI).items():
            s[f"{k} | {cond}"] = v
    if cifake:
        y = np.array([lab for _, lab in cifake])
        p = det.predict([im for im, _ in cifake])
        n += len(y)
        for k, v in metrics(y, p).items():
            s[f"cifake_{k}"] = v
    s["load_s"] = round(det.load_seconds, 1)
    s["ms_per_image_cpu"] = round(1000 * (time.time() - t0) / max(n, 1), 1)
    return per_image, s


def pct(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"


def model_report(s, per_image, source):
    L = [f"# {s['model']} — `{s['repo']}`", "", s["description"], "", f"* Test set: {source}",
         "* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI",
         f"* Load time {s['load_s']} s, inference {s['ms_per_image_cpu']} ms/image on CPU", "",
         "## AI vs real, per perturbation", "", "| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |",
         "|---|---|---|---|---|"]
    for c in HF_CONDITIONS:
        L.append(f"| {c} | {pct(s.get(f'acc | {c}'))} | {pct(s.get(f'tpr | {c}'))} | "
                 f"{pct(s.get(f'fpr | {c}'))} | {s.get(f'auc | {c}', float('nan')):.3f} |")
    if "cifake_acc" in s:
        L += ["", "## CIFAKE test sample (32x32 images)", "",
              f"accuracy {s['cifake_acc']:.1%} · AUC {s['cifake_auc']:.3f} · "
              f"TPR {s['cifake_tpr']:.1%} · FPR {s['cifake_fpr']:.1%}"]
    piv = per_image.pivot_table(index=["label", "image"], columns="condition", values="P_AI")[list(HF_CONDITIONS)]
    L += ["", "## P(AI) per image", "", "| Truth | Image | " + " | ".join(piv.columns) + " |",
          "|---" * (len(piv.columns) + 2) + "|"]
    for (lab, img), r in piv.sort_index(ascending=[True, True]).iterrows():
        L.append(f"| {lab} | {img} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    return "\n".join(L) + "\n"


def cifake_model_wrappers():
    """Your models from ./models, wrapped like hf_detectors.Detector."""
    import detector

    models, _ = detector.load_models()
    wrappers = []
    for m in models:
        def p_ai(imgs, m=m):
            views = [detector.make_views(im)[0] for im in imgs]
            p = detector._p_positive(m, detector.model_inputs(m, views)).astype(float)
            return p if m.settings["positive_class"].upper() == "FAKE" else 1 - p

        wrappers.append(type("W", (), dict(key=f"{m.label}", repo=f"models/{m.name}",
                                           desc="your CIFAKE baseline (32x32 input)", load_seconds=0.0,
                                           predict=staticmethod(p_ai))))
    return wrappers


def leaderboard(summaries, source):
    df = pd.DataFrame(summaries).sort_values(["acc | original", "auc | original"], ascending=False)
    df.to_csv(HF_OUT / "leaderboard.csv", index=False)
    conds = list(HF_CONDITIONS)
    L = ["# Leaderboard — AI vs real", "", f"Test set: {source}. Threshold 0.5. Sorted by accuracy on the original images.", "",
         "| Model | Accuracy | AI caught (TPR) | Real flagged (FPR) | AUC | " +
         " | ".join(f"Acc: {c}" for c in conds[1:]) + " | CIFAKE acc | ms/img |",
         "|---" * (len(conds) + 7) + "|"]
    for _, r in df.iterrows():
        cells = [f"**{r['model']}**", pct(r["acc | original"]), pct(r.get("tpr | original")),
                 pct(r.get("fpr | original")), f"{r['auc | original']:.3f}"]
        cells += [pct(r[f"acc | {c}"]) for c in conds[1:]]
        cells += [pct(r.get("cifake_acc")), f"{r['ms_per_image_cpu']:.0f}"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "| Model | Source | Description |", "|---|---|---|"]
    L += [f"| {r['model']} | `{r['repo']}` | {r['description']} |" for _, r in df.iterrows()]
    (HF_OUT / "leaderboard.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(hf_detectors.REGISTRY), help="comma-separated keys")
    ap.add_argument("--ai-dir", default=str(OUT / "images" / "pollinations"))
    ap.add_argument("--real-dir", default=str(OUT / "images" / "real"))
    ap.add_argument("--cifake", help="CIFAKE zip/folder for the in-domain check (optional)")
    ap.add_argument("--cifake-per-class", type=int, default=100)
    ap.add_argument("--skip-cifake-models", action="store_true", help="don't evaluate ./models")
    args = ap.parse_args()

    items = load_dir(Path(args.ai_dir), 1) + load_dir(Path(args.real_dir), 0)
    n_ai = sum(y for _, y, _ in items)
    source = f"{n_ai} AI images (`{Path(args.ai_dir).name}`) + {len(items) - n_ai} real photos (`{Path(args.real_dir).name}`)"
    print("Test set:", source)
    cifake = load_cifake_sample(args.cifake, args.cifake_per_class) if args.cifake else None
    HF_OUT.mkdir(parents=True, exist_ok=True)

    dets = [k.strip() for k in args.models.split(",") if k.strip()]
    summaries = []
    for key in dets:
        print(f"\n=== {key}: {hf_detectors.REGISTRY[key]['repo']}", flush=True)
        try:
            det = hf_detectors.load_detector(key)
            per_image, s = evaluate_model(det, items, cifake)
        except Exception:
            traceback.print_exc()
            continue
        d = HF_OUT / key
        d.mkdir(exist_ok=True)
        per_image.to_csv(d / "per_image.csv", index=False)
        (d / "report.md").write_text(model_report(s, per_image, source), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        summaries.append(s)
        print(f"  acc {s['acc | original']:.0%} | AI caught {pct(s.get('tpr | original'))} | "
              f"real flagged {pct(s.get('fpr | original'))} | AUC {s['auc | original']:.3f}")
        del det

    if not args.skip_cifake_models:
        print("\n=== your CIFAKE models")
        for w in cifake_model_wrappers():
            per_image, s = evaluate_model(w, items, cifake)
            d = HF_OUT / ("cifake_" + Path(w.repo).stem)
            d.mkdir(exist_ok=True)
            per_image.to_csv(d / "per_image.csv", index=False)
            (d / "report.md").write_text(model_report(s, per_image, source), encoding="utf-8")
            summaries.append(s)

    # keep detectors evaluated in earlier runs on the leaderboard
    done = {s["model"] for s in summaries}
    for f in sorted(HF_OUT.glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s["model"] not in done and s["model"] in hf_detectors.REGISTRY and "acc | original" in s:
            summaries.append(s)
    print("\n" + leaderboard(summaries, source))


if __name__ == "__main__":
    main()
