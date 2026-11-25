"""
Train real-vs-AI detectors on the CIFAKE dataset and save them to ./models

Models trained:
  * pixels_cnn.keras              - Keras CNN (32x32 RGB input, pixels scaled to [0,1])
  * pixels_random_forest.joblib   - scikit-learn RandomForest on flattened pixels
  * pixels_xgboost.joblib         - XGBoost on flattened pixels

OPTIONAL: you already have your own baseline models (cifake_baselines.ipynb). This script
only adds extra raw-pixel models next to them; it never overwrites cnn.keras,
random_forest.joblib or xgboost.json.

Label convention (same as Keras' alphabetical folder ordering):
  class 0 = FAKE, class 1 = REAL  ->  model output / predict_proba[:, 1] = P(REAL)

Usage:
  python train_models.py                                  # downloads CIFAKE with kagglehub
  python train_models.py --data-dir "D:/datasets/cifake"  # use a copy you already have
  python train_models.py --epochs 3 --classic-samples 5000   # quick run
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 32
CLASS_NAMES = ["FAKE", "REAL"]  # index 0 = FAKE, index 1 = REAL
KAGGLE_DATASET = "birdy654/cifake-real-and-ai-generated-synthetic-images"
MODELS_DIR = Path(__file__).parent / "models"


# --------------------------------------------------------------------------- data
def find_cifake_root(start: Path) -> Path:
    """Return the folder that contains train/{REAL,FAKE} and test/{REAL,FAKE}."""
    candidates = [start] + [p for p in start.rglob("*") if p.is_dir()]
    for c in candidates:
        if all((c / split / cls).is_dir() for split in ("train", "test") for cls in CLASS_NAMES):
            return c
    raise FileNotFoundError(
        f"Could not find train/REAL, train/FAKE, test/REAL, test/FAKE under {start}"
    )


def get_data_root(data_dir: str | None) -> Path:
    if data_dir:
        return find_cifake_root(Path(data_dir).expanduser().resolve())
    print("No --data-dir given -> downloading CIFAKE from Kaggle via kagglehub ...")
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    return find_cifake_root(Path(path))


def load_numpy_subset(split_dir: Path, per_class: int, seed: int = 0):
    """Load up to `per_class` images per class into (N, 32*32*3) float32 in [0,1]."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, cls in enumerate(CLASS_NAMES):
        files = sorted(
            f for f in (split_dir / cls).iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if per_class and per_class < len(files):
            files = list(rng.choice(files, size=per_class, replace=False))
        for f in files:
            img = Image.open(f).convert("RGB")
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
            X.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
            y.append(label)
    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# --------------------------------------------------------------------------- CNN
def build_cnn():
    import tensorflow as tf
    from tensorflow.keras import layers

    aug = tf.keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomTranslation(0.05, 0.05)],
        name="augment",
    )

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return layers.MaxPooling2D()(x)

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))  # expects pixels already scaled to [0,1]
    x = aug(inp)
    x = block(x, 32)
    x = block(x, 64)
    x = block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_real")(x)
    model = tf.keras.Model(inp, out, name="cifake_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_cnn(root: Path, epochs: int, batch: int) -> dict:
    import tensorflow as tf

    common = dict(
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch,
        label_mode="binary",
        class_names=CLASS_NAMES,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root / "train", validation_split=0.1, subset="training", seed=42, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root / "train", validation_split=0.1, subset="validation", seed=42, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(root / "test", shuffle=False, **common)

    scale = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(scale, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(scale, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(scale, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    model = build_cnn()
    model.summary()
    out_path = MODELS_DIR / "pixels_cnn.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(out_path)
    loss, acc, auc = model.evaluate(test_ds, verbose=0)
    print(f"[CNN] test accuracy={acc:.4f}  auc={auc:.4f}  -> {out_path}")
    return {"accuracy": float(acc), "auc": float(auc)}


# --------------------------------------------------------------------------- classic ML
def train_classic(root: Path, per_class: int, skip_xgb: bool) -> dict:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    print(f"Loading {per_class} train images per class for classic models ...")
    X_tr, y_tr = load_numpy_subset(root / "train", per_class)
    X_te, y_te = load_numpy_subset(root / "test", max(per_class // 5, 1000), seed=1)
    print(f"  train {X_tr.shape}, test {X_te.shape}")

    results = {}
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42, min_samples_leaf=2
        )
    }
    if not skip_xgb:
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.5, tree_method="hist", n_jobs=-1, eval_metric="logloss",
            )
        except ImportError:
            print("xgboost not installed -> skipping XGBoost")

    for name, clf in models.items():
        t = time.time()
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, (p >= 0.5).astype(int))
        auc = roc_auc_score(y_te, p)
        path = MODELS_DIR / f"pixels_{name}.joblib"
        joblib.dump(clf, path)
        print(f"[{name}] test accuracy={acc:.4f}  auc={auc:.4f}  ({time.time() - t:.0f}s) -> {path}")
        results[name] = {"accuracy": float(acc), "auc": float(auc)}
    return results


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="CIFAKE folder containing train/ and test/ (skips download)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--classic-samples", type=int, default=15000,
                    help="images PER CLASS used for RandomForest/XGBoost (RAM/time trade-off)")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--skip-classic", action="store_true")
    ap.add_argument("--skip-xgb", action="store_true")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    root = get_data_root(args.data_dir)
    print(f"Using CIFAKE at: {root}")

    metrics_path = MODELS_DIR / "pixels_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if not args.skip_cnn:
        metrics["pixels_cnn"] = train_cnn(root, args.epochs, args.batch_size)
    if not args.skip_classic:
        metrics.update(train_classic(root, args.classic_samples, args.skip_xgb))
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Per-model settings read by the Streamlit app (see detector.py).
    cfg_path = MODELS_DIR / "models.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    for f in ("pixels_cnn.keras", "pixels_random_forest.joblib", "pixels_xgboost.joblib"):
        if (MODELS_DIR / f).exists():
            cfg.setdefault(f, {"positive_class": "REAL", "scale_0_1": True})
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"\nDone. Metrics: {metrics_path}\nNow run:  streamlit run app.py")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
