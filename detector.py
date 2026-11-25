"""
Load trained real-vs-AI detectors from ./models and run them on any image.

Supported model files:
  *.keras / *.h5      Keras / TensorFlow models (CNNs)
  *.joblib / *.pkl    scikit-learn compatible models with predict_proba
  *.json              XGBoost native models (only if listed in models.json with "loader": "xgboost")

Per-model settings live in models/models.json, e.g. (this is the setup for the CIFAKE
baseline notebook, where REAL=0 and FAKE=1):
  {
    "cnn.keras":            {"label": "CNN", "positive_class": "FAKE", "scale_0_1": false},
    "random_forest.joblib": {"label": "Random Forest", "positive_class": "FAKE", "features": "handcrafted"},
    "xgboost.json":         {"label": "XGBoost", "positive_class": "FAKE", "features": "handcrafted",
                             "loader": "xgboost"}
  }
  label          : display name in the app (default: file name)
  positive_class : which class the model's output / predict_proba[:, 1] refers to (default "REAL")
  scale_0_1      : divide pixels by 255 before feeding the model (default true). Use false when
                   the model has a Rescaling(1/255) layer inside it.
  features       : "pixels" (flattened raw pixels, default) or "handcrafted" (calls
                   extract_features(uint8 HxWx3) from models/features.py, or "features_file")
  loader         : "keras" | "joblib" | "xgboost" (default: from the file extension)
  img_size       : input side for handcrafted-feature models (default 32)
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
KERAS_EXT = {".keras", ".h5"}
JOBLIB_EXT = {".joblib", ".pkl"}
DEFAULT_SETTINGS = {"positive_class": "REAL", "scale_0_1": True, "features": "pixels"}
RESAMPLE = Image.BILINEAR  # same as the training notebook's resize


@dataclass
class LoadedModel:
    name: str  # file name
    label: str  # display name
    kind: str  # "keras" | "sklearn"
    obj: object
    size: int  # square input side the model expects, e.g. 32
    channels: int  # 3 = RGB, 1 = grayscale
    settings: dict = field(default_factory=dict)
    extractor: Callable[[np.ndarray], np.ndarray] | None = None  # handcrafted features


@dataclass
class Prediction:
    model: str
    p_fake: float  # averaged over all views
    per_view: list[float]

    @property
    def verdict(self) -> str:
        return "AI-GENERATED" if self.p_fake >= 0.5 else "REAL"

    @property
    def confidence(self) -> float:
        return max(self.p_fake, 1 - self.p_fake)


# --------------------------------------------------------------------------- loading
def _read_settings(models_dir: Path) -> dict:
    cfg = models_dir / "models.json"
    return json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}


_feature_modules: dict[Path, object] = {}


def _feature_extractor(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    path = path.resolve()
    if path not in _feature_modules:
        if not path.exists():
            raise FileNotFoundError(f"feature file {path.name} not found in {path.parent}")
        spec = importlib.util.spec_from_file_location(f"features_{len(_feature_modules)}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _feature_modules[path] = mod
    return _feature_modules[path].extract_features


def _load_one(path: Path, settings: dict, models_dir: Path) -> LoadedModel:
    s = {**DEFAULT_SETTINGS, **settings}
    ext = path.suffix.lower()
    loader = s.get("loader") or ("keras" if ext in KERAS_EXT else "xgboost" if ext == ".json" else "joblib")
    label = s.get("label", path.name)

    if loader == "keras":
        import tensorflow as tf

        model = tf.keras.models.load_model(path, compile=False)
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        _, h, w, c = shape
        return LoadedModel(path.name, label, "keras", model, int(h or 32), int(c or 3), s)

    if loader == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(path)
    else:
        import warnings

        import joblib

        with warnings.catch_warnings():  # models pickled with Colab's sklearn version
            warnings.simplefilter("ignore")
            model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError("model has no predict_proba()")

    if s["features"] == "handcrafted":
        extractor = _feature_extractor(models_dir / s.get("features_file", "features.py"))
        return LoadedModel(path.name, label, "sklearn", model, int(s.get("img_size", 32)), 3, s, extractor)

    n = int(getattr(model, "n_features_in_", 32 * 32 * 3))
    if n % 3 == 0 and math.isqrt(n // 3) ** 2 == n // 3:
        size, ch = math.isqrt(n // 3), 3
    elif math.isqrt(n) ** 2 == n:
        size, ch = math.isqrt(n), 1
    else:
        raise ValueError(
            f"expects {n} features, which isn't a raw-pixel input. If it uses hand-crafted "
            'features, set "features": "handcrafted" for it in models/models.json.'
        )
    return LoadedModel(path.name, label, "sklearn", model, size, ch, s)


def load_models(models_dir: Path = MODELS_DIR) -> tuple[list[LoadedModel], list[str]]:
    """Load every supported model file. Returns (models, error_messages)."""
    if not models_dir.exists():
        return [], [f"Folder {models_dir} does not exist."]
    settings = _read_settings(models_dir)
    models, errors = [], []
    for path in sorted(models_dir.iterdir()):
        ext = path.suffix.lower()
        if ext == ".json":  # models.json / meta.json etc. are not models
            if path.name not in settings:
                continue
        elif ext not in KERAS_EXT | JOBLIB_EXT:
            continue
        try:
            models.append(_load_one(path, settings.get(path.name, {}), models_dir))
        except Exception as e:  # keep loading the others
            errors.append(f"{path.name}: {type(e).__name__}: {e}")
    return models, errors


# --------------------------------------------------------------------------- preprocessing
def make_views(img: Image.Image, n_crops: int = 0, seed: int = 0) -> list[Image.Image]:
    """View 0 = centre square crop of the whole image; then `n_crops` random square crops
    (50-100% of the short side). CIFAKE images are square 32x32, so we keep square inputs
    instead of squashing the aspect ratio."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    views = [img.crop((left, top, left + s, top + s))]
    rng = random.Random(seed)
    for _ in range(n_crops):
        side = max(8, int(s * rng.uniform(0.5, 1.0)))
        x, y = rng.randint(0, w - side), rng.randint(0, h - side)
        views.append(img.crop((x, y, x + side, y + side)))
    return views


def to_uint8(views: list[Image.Image], size: int, channels: int) -> np.ndarray:
    mode = "RGB" if channels == 3 else "L"
    arr = np.stack([np.asarray(v.convert(mode).resize((size, size), RESAMPLE), dtype=np.uint8) for v in views])
    return arr[..., None] if channels == 1 else arr


def model_inputs(m: LoadedModel, views: list[Image.Image]) -> np.ndarray:
    u8 = to_uint8(views, m.size, m.channels)
    if m.extractor is not None:
        return np.stack([m.extractor(a) for a in u8])
    x = u8.astype(np.float32)
    if m.settings["scale_0_1"]:
        x /= 255.0
    return x if m.kind == "keras" else x.reshape(len(x), -1)


# --------------------------------------------------------------------------- inference
def _p_positive(m: LoadedModel, x: np.ndarray) -> np.ndarray:
    if m.kind == "keras":
        out = np.asarray(m.obj.predict(x, verbose=0))
        if out.ndim == 1 or out.shape[-1] == 1:  # sigmoid
            return out.reshape(-1)
        return out[:, 1]  # softmax over 2 classes
    proba = m.obj.predict_proba(x)
    classes = list(getattr(m.obj, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else 1
    return proba[:, idx]


def predict(models: list[LoadedModel], img: Image.Image, n_crops: int = 0) -> list[Prediction]:
    views = make_views(img, n_crops)
    preds = []
    for m in models:
        p_pos = _p_positive(m, model_inputs(m, views)).astype(float)
        p_fake = p_pos if m.settings["positive_class"].upper() == "FAKE" else 1.0 - p_pos
        preds.append(Prediction(m.label, float(np.mean(p_fake)), p_fake.tolist()))
    return preds


def ensemble(preds: list[Prediction]) -> Prediction | None:
    if not preds:
        return None
    p = float(np.mean([x.p_fake for x in preds]))
    return Prediction("Ensemble (mean)", p, [x.p_fake for x in preds])


def model_input_preview(img: Image.Image, size: int = 32) -> Image.Image:
    """What a 32x32 model actually sees, upscaled with NEAREST so the pixels are visible."""
    small = make_views(img)[0].resize((size, size), RESAMPLE)
    return small.resize((size * 8, size * 8), Image.NEAREST)
