"""
AI-Generated Image Detector - Streamlit app

  1. Generate an image from a text prompt (Gemini API, or a free keyless fallback)
  2. Or upload any image (a real photo or an AI image)
  3. Community Forensics (pretrained, best open-source detector in our evaluation) and every
     CIFAKE model in ./models predict REAL vs AI-GENERATED with a confidence score

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st
from PIL import Image

import detector
import generator

st.set_page_config(page_title="AI Image Detector", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------- helpers
def secret(name: str) -> str:
    """.streamlit/secrets.toml first, then env vars / the project's .env file."""
    try:
        val = st.secrets.get(name, "")
    except Exception:  # no secrets.toml
        val = ""
    return val or generator.load_api_key()


def models_signature() -> tuple:
    d = detector.MODELS_DIR
    if not d.exists():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime) for p in d.iterdir()))


@st.cache_resource(show_spinner="Loading detection models ...")
def get_models(_sig: tuple):
    return detector.load_models()


COMMFOR_LABEL = "⭐ Community Forensics (pretrained)"


@st.cache_resource(show_spinner="Loading Community Forensics (first run downloads ~90 MB) ...")
def get_commfor():
    """Returns (detector, error message)."""
    try:
        import hf_detectors

        return hf_detectors.load_detector("commfor"), None
    except Exception as e:  # torch/timm not installed, or offline on first run
        return None, f"{type(e).__name__}: {e}"


def init_state():
    st.session_state.setdefault("generated", None)  # (PIL image, prompt, backend)
    st.session_state.setdefault("history", [])


# --------------------------------------------------------------------------- sidebar
init_state()
st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("Image generation")
backend = st.sidebar.radio(
    "Backend",
    ["Gemini API", "Pollinations (free, no key)"],
    help="Gemini's free tier may have no quota for image models. Pollinations is a keyless fallback "
    "so you can still test the pipeline.",
)
api_key = ""
gemini_model = generator.GEMINI_IMAGE_MODELS[0]
if backend == "Gemini API":
    api_key = st.sidebar.text_input(
        "Gemini API key", value=secret("GEMINI_API_KEY"), type="password",
        help="Get one at https://aistudio.google.com/apikey . Can also be set via the "
        "GEMINI_API_KEY env var or .streamlit/secrets.toml",
    )
    options = st.session_state.get("gemini_models") or generator.GEMINI_IMAGE_MODELS
    gemini_model = st.sidebar.selectbox("Gemini image model", options + ["(custom)"])
    if gemini_model == "(custom)":
        gemini_model = st.sidebar.text_input("Custom model id", "gemini-2.5-flash-image")
    if st.sidebar.button("List image models for my key", disabled=not api_key):
        try:
            found = generator.list_gemini_image_models(api_key)
            st.session_state["gemini_models"] = found or generator.GEMINI_IMAGE_MODELS
            st.sidebar.success(f"Found: {', '.join(found) or 'none'}")
            st.rerun()
        except Exception as e:
            st.sidebar.error(str(e))

st.sidebar.subheader("Detection")
use_commfor = st.sidebar.checkbox(
    "Use Community Forensics (pretrained)", value=True,
    help="ViT-S/16 at 384 px from OwensLab/commfor-model-384 (Park & Owens), trained on images "
    "from 4,803 generators. Best open-source detector in our evaluation. Works on the full-resolution "
    "image instead of a 32x32 thumbnail.",
)
commfor, commfor_err = get_commfor() if use_commfor else (None, None)
verdict_options = ([COMMFOR_LABEL] if commfor else []) + ["CIFAKE models (ensemble)"]
verdict_from = st.sidebar.radio("Final verdict from", verdict_options)
n_crops = st.sidebar.slider(
    "Extra random crops (test-time augmentation)", 0, 16, 4,
    help="0 = only the whole image (centre-cropped to a square, resized to the model input). "
    "More crops -> predictions averaged over several zoomed-in views, which helps a little on "
    "high-resolution images.",
)
threshold = st.sidebar.slider("Decision threshold on P(AI)", 0.05, 0.95, 0.50, 0.05)

if st.sidebar.button("🔄 Reload models"):
    get_models.clear()
models, load_errors = get_models(models_signature())
st.sidebar.subheader(f"Loaded models ({len(models) + bool(commfor)})")
if commfor:
    st.sidebar.caption(f"✅ **Community Forensics** ({commfor.repo}) · 384x384 pixels")
elif use_commfor:
    st.sidebar.error(f"Community Forensics not available: {commfor_err}\n\n"
                     "Install: pip install -r requirements.txt (needs torch, timm).")
for m in models:
    inp = "handcrafted features" if m.extractor else f"{m.size}x{m.size}x{m.channels} pixels"
    st.sidebar.caption(f"✅ **{m.label}** ({m.name}) · {inp}")
for err in load_errors:
    st.sidebar.error(err)


# --------------------------------------------------------------------------- analysis view
def analyse(img: Image.Image, source: str, truth: str, key: str):
    if not models and not commfor:
        st.warning("No detectors available - check the sidebar.")
        return
    with st.spinner("Running detectors ..."):
        preds = detector.predict(models, img, n_crops=n_crops) if models else []
        ens = detector.ensemble(preds)
        p_cf = float(commfor.predict([img])[0]) if commfor else None

    if verdict_from == COMMFOR_LABEL and p_cf is not None:
        p_final, final_name = p_cf, "Community Forensics"
    else:
        p_final, final_name = ens.p_fake, "CIFAKE ensemble"
    is_ai = p_final >= threshold

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.image(img, caption=f"{source} · {img.width}x{img.height}", width="stretch")
    with c2:
        st.image(detector.model_input_preview(img), caption="What a 32x32 CIFAKE model sees",
                 width="stretch")
    with c3:
        if is_ai:
            st.error(f"### 🤖 AI-GENERATED\nP(AI) = **{p_final:.1%}** · {final_name}")
        else:
            st.success(f"### 📷 REAL\nP(real) = **{1 - p_final:.1%}** · {final_name}")
        m1, m2 = st.columns(2)
        if p_cf is not None:
            m1.metric("Community Forensics P(AI)", f"{p_cf:.1%}")
        if ens is not None:
            m2.metric("CIFAKE ensemble P(AI)", f"{ens.p_fake:.1%}")
        st.progress(p_final, text=f"Final P(AI-generated) = {p_final:.3f}")
        if truth != "Unknown":
            correct = (truth == "AI-generated") == is_ai
            st.write(("✅ Correct" if correct else "❌ Wrong") + f" (ground truth: {truth})")

    rows = []
    if p_cf is not None:
        rows.append({"Model": COMMFOR_LABEL, "P(AI)": p_cf, "Input": "384x384 full-res crop",
                     "Min/Max over views": "-"})
    rows += [{"Model": p.model, "P(AI)": p.p_fake, "Input": "32x32",
              "Min/Max over views": f"{min(p.per_view):.2f} / {max(p.per_view):.2f}"} for p in preds]
    if ens is not None:
        rows.append({"Model": "CIFAKE ensemble (mean)", "P(AI)": ens.p_fake, "Input": "32x32",
                     "Min/Max over views": "-"})
    for r in rows:
        r["Verdict"] = "AI-GENERATED" if r["P(AI)"] >= threshold else "REAL"
        r["Confidence"] = f"{max(r['P(AI)'], 1 - r['P(AI)']):.1%}"
    df = pd.DataFrame(rows)[["Model", "P(AI)", "Verdict", "Confidence", "Input", "Min/Max over views"]]
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "P(AI)": st.column_config.ProgressColumn("P(AI)", min_value=0, max_value=1, format="%.3f"),
        },
    )

    if st.button("➕ Add result to evaluation log", key=f"log_{key}"):
        st.session_state.history.append({
            "Source": source, "Ground truth": truth, "Verdict from": final_name,
            "P(AI)": round(p_final, 4), "Verdict": "AI-GENERATED" if is_ai else "REAL",
            **({"Community Forensics": round(p_cf, 4)} if p_cf is not None else {}),
            **({"CIFAKE ensemble": round(ens.p_fake, 4)} if ens is not None else {}),
            **{p.model: round(p.p_fake, 4) for p in preds},
        })
        st.toast("Logged")


# --------------------------------------------------------------------------- main page
st.title("🔍 AI-Generated Image Detector")
st.caption("Phase 2 pipeline: prompt → generated image → detector verdict + confidence. "
           "Compares the pretrained Community Forensics detector (works at 384 px) with your "
           "CIFAKE models (trained on 32x32 images, so high-resolution inputs are shrunk - see "
           "'What a 32x32 CIFAKE model sees').")

tab_gen, tab_up, tab_log = st.tabs(["✨ Generate from prompt", "📤 Upload an image", "📊 Evaluation log"])

with tab_gen:
    prompt = st.text_area("Prompt", "A photorealistic photo of a golden retriever sitting on a park bench at sunset")
    if st.button("Generate image", type="primary", disabled=not prompt.strip()):
        try:
            with st.spinner(f"Generating with {backend} ..."):
                if backend == "Gemini API":
                    if not api_key:
                        raise generator.GenerationError("Enter your Gemini API key in the sidebar.")
                    img = generator.generate_gemini(api_key, prompt, gemini_model)
                    label = f"Gemini ({gemini_model})"
                else:
                    img = generator.generate_pollinations(prompt)
                    label = "Pollinations"
            st.session_state.generated = (img, prompt, label)
        except generator.GenerationError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)

    if st.session_state.generated:
        img, used_prompt, label = st.session_state.generated
        st.caption(f"Prompt: _{used_prompt}_")
        analyse(img, label, "AI-generated", "gen")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button("⬇️ Download generated image", buf.getvalue(), "generated.png", "image/png")

with tab_up:
    up = st.file_uploader("Upload a real photo or an AI-generated image",
                          type=["png", "jpg", "jpeg", "webp", "bmp"])
    truth = st.radio("Ground truth (optional, for the evaluation log)",
                     ["Unknown", "Real", "AI-generated"], horizontal=True)
    if up is not None:
        analyse(Image.open(up).convert("RGB"), f"Upload: {up.name}", truth, "up")

with tab_log:
    hist = st.session_state.history
    if not hist:
        st.info("Click 'Add result to evaluation log' under any prediction to collect results here.")
    else:
        df = pd.DataFrame(hist)
        labelled = df[df["Ground truth"] != "Unknown"]
        if len(labelled):
            acc = ((labelled["Ground truth"] == "AI-generated") == (labelled["Verdict"] == "AI-GENERATED")).mean()
            st.metric("Accuracy of the final verdict on labelled samples", f"{acc:.1%}",
                      help=f"{len(labelled)} samples")
        st.dataframe(df, width="stretch")
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "evaluation_log.csv", "text/csv")
        if st.button("Clear log"):
            st.session_state.history = []
            st.rerun()
