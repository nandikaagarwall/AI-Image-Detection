"""
Evaluate the CIFAKE-trained detectors on freshly generated (AI) images.

What it does
  1. Generates N images (default 10), one per CIFAR-10 class, so content matches CIFAKE.
  2. Runs every model in ./models on each image under several conditions:
       original, original+4 random crops (TTA), JPEG q=75, downscale 50%,
       Gaussian noise (sigma=8), centre crop 70%
  3. Optional in-domain reference: CIFAKE test images (REAL + FAKE) from the dataset
     zip/folder -> accuracy/AUC there, plus AUC of "generated fakes vs CIFAKE reals".
  4. Writes evaluation/report.md, evaluation/per_image.csv, evaluation/summary.csv,
     evaluation/contact_sheet.png

Usage
  python evaluate.py --backend gemini --gemini-model gemini-2.5-flash-image
  python evaluate.py --backend pollinations
  python evaluate.py --backend pollinations --cifake "C:/Users/you/Downloads/nandika_project.zip"
  python evaluate.py --images-dir my_ai_images      # skip generation, evaluate a folder of AI images
"""
from __future__ import annotations

import argparse
import io
import random
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import detector
import generator

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
OUT = ROOT / "evaluation"

PROMPTS = [  # one per CIFAR-10 class -> same semantic content as CIFAKE
    ("airplane", "A photorealistic photograph of a passenger airplane taking off from a runway, daylight"),
    ("automobile", "A photorealistic photograph of a red sedan car parked on a city street"),
    ("bird", "A photorealistic close-up photograph of a small sparrow perched on a branch"),
    ("cat", "A photorealistic photograph of a tabby cat sitting on a windowsill"),
    ("deer", "A photorealistic photograph of a deer standing in a forest clearing"),
    ("dog", "A photorealistic photograph of a golden retriever playing on grass"),
    ("frog", "A photorealistic macro photograph of a green frog on a lily pad"),
    ("horse", "A photorealistic photograph of a brown horse running in a field"),
    ("ship", "A photorealistic photograph of a large cargo ship at sea"),
    ("truck", "A photorealistic photograph of a delivery truck on a highway"),
]


# --------------------------------------------------------------------------- perturbations
def jpeg(img, q=75):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def downscale(img, f=0.5):
    return img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)


def noise(img, sigma=8, seed=0):
    a = np.asarray(img, dtype=np.float32)
    a = a + np.random.default_rng(seed).normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def center_crop(img, f=0.7):
    w, h = img.size
    cw, ch = int(w * f), int(h * f)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


CONDITIONS = {  # name -> (transform, n_crops)
    "original": (lambda im: im, 0),
    "original + TTA(4 crops)": (lambda im: im, 4),
    "JPEG q=75": (jpeg, 0),
    "downscale 50%": (downscale, 0),
    "noise sigma=8": (noise, 0),
    "centre crop 70%": (center_crop, 0),
}


# --------------------------------------------------------------------------- data
def generate_images(args) -> list[tuple[str, str, Path]]:
    img_dir = OUT / "images" / args.backend
    img_dir.mkdir(parents=True, exist_ok=True)
    key = generator.load_api_key() if args.backend == "gemini" else ""
    if args.backend == "gemini" and not key:
        raise SystemExit("No Gemini key found (.env / GEMINI_API_KEY).")
    items = []
    for i, (cls, prompt) in enumerate(PROMPTS[: args.n]):
        path = img_dir / f"{i:02d}_{cls}.png"
        if not path.exists():  # cached -> re-runs don't spend quota again
            for attempt in range(3):
                try:
                    print(f"  generating {path.name} ...", flush=True)
                    if args.backend == "gemini":
                        img = generator.generate_gemini(key, prompt, args.gemini_model)
                    else:
                        img = generator.generate_pollinations(prompt, seed=1000 + i)
                    img.save(path)
                    break
                except generator.GenerationError as e:
                    if "limit: 0" in str(e):
                        raise SystemExit(f"\n{e}")
                    print(f"    attempt {attempt + 1} failed: {str(e)[:150]}")
                    time.sleep(max(args.delay, 10) * (attempt + 1))
            else:
                print(f"  skipping {cls}: generation failed 3 times")
                continue
            time.sleep(args.delay)
        items.append((cls, prompt, path))
    return items


def load_cifake_sample(src: str, per_class: int, seed: int = 0):
    """Returns list of (PIL image, label) with label 1 = FAKE, from test/REAL + test/FAKE."""
    rng = random.Random(seed)
    out = []
    p = Path(src)
    if p.suffix.lower() == ".zip":
        z = zipfile.ZipFile(p)
        groups = {"REAL": [], "FAKE": []}
        for n in z.namelist():
            parts = n.replace("\\", "/").split("/")
            if len(parts) >= 3 and parts[-3].lower() == "test" and parts[-2].upper() in groups \
                    and parts[-1].lower().endswith((".jpg", ".jpeg", ".png")):
                groups[parts[-2].upper()].append(n)
        for cls, names in groups.items():
            for n in rng.sample(names, min(per_class, len(names))):
                out.append((Image.open(io.BytesIO(z.read(n))).convert("RGB"), int(cls == "FAKE")))
    else:
        for cls in ("REAL", "FAKE"):
            d = next(p.rglob(f"test/{cls}"), None) or p / cls
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            for f in rng.sample(files, min(per_class, len(files))):
                out.append((Image.open(f).convert("RGB"), int(cls == "FAKE")))
    return out


# --------------------------------------------------------------------------- metrics
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")


def batch_p_fake(models, imgs):
    """Fast path: P(FAKE) for many images at once (whole image, no crops)."""
    out = {}
    for m in models:
        views = [detector.make_views(im)[0] for im in imgs]
        p = detector._p_positive(m, detector.model_inputs(m, views)).astype(float)
        out[m.label] = p if m.settings["positive_class"].upper() == "FAKE" else 1 - p
    out["Ensemble"] = np.mean([out[m.label] for m in models], axis=0)
    return out


def contact_sheet(items, per_image, path, thumb=180):
    cols = 5
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 34)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (cls, _, p) in enumerate(items):
        im = Image.open(p).convert("RGB")
        im = detector.make_views(im)[0].resize((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 34)
        sheet.paste(im, (x, y))
        pf = per_image[(per_image.image == p.name) & (per_image.condition == "original")]["Ensemble"].iloc[0]
        d.text((x + 4, y + thumb + 2), f"{cls}  P(AI)={pf:.2f}", fill="black")
        d.text((x + 4, y + thumb + 16), "DETECTED" if pf >= 0.5 else "missed", fill="red" if pf >= 0.5 else "gray")
    sheet.save(path)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["gemini", "pollinations"], default="gemini")
    ap.add_argument("--gemini-model", default=generator.GEMINI_IMAGE_MODELS[0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--delay", type=float, default=None,
                    help="seconds between generation calls (default: 50 for pollinations, "
                         "whose free tier allows 1 image / 45 s per IP; 5 for gemini)")
    ap.add_argument("--images-dir", help="evaluate existing AI images instead of generating")
    ap.add_argument("--cifake", help="CIFAKE zip or folder for the in-domain reference")
    ap.add_argument("--cifake-per-class", type=int, default=300)
    args = ap.parse_args()
    if args.delay is None:
        args.delay = 50.0 if args.backend == "pollinations" else 5.0

    OUT.mkdir(exist_ok=True)
    models, errors = detector.load_models()
    if errors:
        print("Model load errors:", errors)
    labels = [m.label for m in models] + ["Ensemble"]
    print("Models:", [m.label for m in models])

    if args.images_dir:
        files = sorted(f for f in Path(args.images_dir).iterdir()
                       if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        items = [(f.stem, "", f) for f in files]
        source = f"folder {args.images_dir}"
    else:
        print(f"Generating {args.n} images with {args.backend} ...")
        items = generate_images(args)
        source = f"{args.backend}" + (f" ({args.gemini_model})" if args.backend == "gemini" else "")
    if not items:
        raise SystemExit("No images to evaluate.")

    # ---- generated images x conditions
    rows = []
    for cls, prompt, path in items:
        img = Image.open(path).convert("RGB")
        for cond, (fn, n_crops) in CONDITIONS.items():
            preds = detector.predict(models, fn(img), n_crops=n_crops)
            row = {"image": path.name, "class": cls, "size": f"{img.width}x{img.height}", "condition": cond}
            row.update({p.model: p.p_fake for p in preds})
            row["Ensemble"] = float(np.mean([p.p_fake for p in preds]))
            rows.append(row)
    per_image = pd.DataFrame(rows)
    per_image.to_csv(OUT / "per_image.csv", index=False)

    summary = per_image.groupby("condition", sort=False)[labels].agg(
        [("detection_rate", lambda s: (s >= 0.5).mean()), ("mean_P_AI", "mean")]
    )
    summary.to_csv(OUT / "summary.csv")

    # ---- in-domain reference + generated-vs-real AUC
    ref_lines = []
    if args.cifake:
        print(f"Loading CIFAKE reference sample from {args.cifake} ...")
        sample = load_cifake_sample(args.cifake, args.cifake_per_class)
        y = np.array([lab for _, lab in sample])
        pr = batch_p_fake(models, [im for im, _ in sample])
        gen_orig = per_image[per_image.condition == "original"]
        reals = y == 0
        ref_lines += [
            f"In-domain CIFAKE test sample: {reals.sum()} real + {(~reals).sum()} fake (32x32).",
            "",
            "| Model | CIFAKE accuracy | CIFAKE AUC | AUC: generated fakes vs CIFAKE reals |",
            "|---|---|---|---|",
        ]
        for lab in labels:
            acc = ((pr[lab] >= 0.5) == y).mean()
            y2 = np.r_[np.zeros(reals.sum()), np.ones(len(gen_orig))]
            p2 = np.r_[pr[lab][reals], gen_orig[lab].to_numpy()]
            ref_lines.append(f"| {lab} | {acc:.1%} | {auc(y, pr[lab]):.3f} | {auc(y2, p2):.3f} |")

    contact_sheet(items, per_image, OUT / "contact_sheet.png")

    # ---- report
    orig = per_image[per_image.condition == "original"]
    L = [
        "# Detector evaluation on generated images",
        "",
        f"* Source of AI images: **{source}**, {len(items)} images, one per CIFAR-10 class",
        f"* Models: {', '.join(m.label for m in models)} (all trained on CIFAKE 32x32)",
        "* Every image is AI-generated, so the ideal output is P(AI) = 1 and **detection rate = recall on fakes**.",
        "* Threshold 0.5. Images are centre-cropped to a square and resized to 32x32 (bilinear), like in training.",
        "",
        "## 1. Detection rate on generated images (original, no perturbation)",
        "",
        "| Model | Detected | Detection rate | Mean P(AI) | Median P(AI) |",
        "|---|---|---|---|---|",
    ]
    for lab in labels:
        s = orig[lab]
        L.append(f"| {lab} | {(s >= .5).sum()}/{len(s)} | {(s >= .5).mean():.0%} | {s.mean():.3f} | {s.median():.3f} |")
    L += ["", "## 2. Robustness: detection rate under perturbations", "",
          "| Condition | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for cond in CONDITIONS:
        sub = per_image[per_image.condition == cond]
        L.append(f"| {cond} | " + " | ".join(f"{(sub[l] >= .5).mean():.0%} ({sub[l].mean():.2f})" for l in labels) + " |")
    L += ["", "_Cell = detection rate (mean P(AI))._", ""]
    if ref_lines:
        L += ["## 3. In-domain reference (CIFAKE test set)", ""] + ref_lines + [""]
    L += ["## Per-image P(AI) (original)", "", "| Image | Size | " + " | ".join(labels) + " |",
          "|---" * (len(labels) + 2) + "|"]
    for _, r in orig.iterrows():
        L.append(f"| {r['image']} | {r['size']} | " + " | ".join(f"{r[l]:.3f}" for l in labels) + " |")
    L += ["", "![contact sheet](contact_sheet.png)", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nSaved: {OUT / 'report.md'}, per_image.csv, summary.csv, contact_sheet.png")


if __name__ == "__main__":
    main()
