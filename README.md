# AI-Generated Image Detector — Phase 2

A Streamlit demo plus an evaluation harness for detecting AI-generated images.

* **Generate** an image from a prompt (Gemini API, or the free Pollinations fallback), or **upload** any image.
* **Detect** it with:
  * ⭐ **Community Forensics** (`OwensLab/commfor-model-384`): a pretrained open-source detector and the best model in our evaluation. It looks at the image at 384×384.
  * **Your CIFAKE baselines**: CNN, Random Forest and XGBoost from `cifake_baselines.ipynb`. They look at a 32×32 thumbnail.
* **Evaluate** detectors on AI images vs real photos, with robustness tests (JPEG, resize, noise, crop).

---

## Project layout

```
app.py                  Streamlit demo
detector.py             loads your CIFAKE models (models/) with the notebook's exact preprocessing
hf_detectors.py         8 pretrained open-source detectors (Community Forensics + 7 Hugging Face models)
generator.py            image generation: Gemini (google-genai) + Pollinations; reads the key from .env
evaluate.py             CIFAKE models on generated images (+ robustness, CIFAKE reference)
evaluate_hf.py          ALL detectors on AI images vs real photos -> leaderboard
fetch_real_photos.py    downloads real photos (Wikimedia Commons, pre-2020) for the evaluation
train_models.py         optional: trains extra raw-pixel CIFAKE models (never overwrites yours)
requirements.txt
.env                    Gemini_Api_Key=...      (private, git-ignored)
models/                 cnn.keras, random_forest.joblib, xgboost.json, features.py, models.json, ...
evaluation/
  images/pollinations/  10 AI-generated test images (one per CIFAR-10 class)
  images/real/          10 real photos + CREDITS.csv (authors and licences)
  report.md             CIFAKE models on the generated images
  open_source/          one folder per detector (report.md, per_image.csv) + leaderboard.md
```

---

## Step 1 — Install (once)

Requirements: Windows, **Python 3.12** (3.10–3.12 work; TensorFlow doesn't support newer versions), about 4 GB of free disk space.

Open **PowerShell** in the project folder. In File Explorer, open the folder, type `powershell` in the address bar and press Enter. Then:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

If you see *"running scripts is disabled on this system"*, run this once and then repeat the activate command:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Your prompt now starts with `(.venv)`. Install the packages (5–10 minutes; TensorFlow and PyTorch are large):

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

> Always work inside `.venv`. Your global Python has a TensorFlow/protobuf version clash that makes `import tensorflow` fail.

## Step 2 — API key (optional)

The key is not shipped with the project. Create it once from the template:

```powershell
Copy-Item .env.example .env
```

```powershell
notepad .env
```

The first command makes a `.env` file from the template. The second opens it so you can paste your key after the `=` (get one at <https://aistudio.google.com/apikey>):

```
Gemini_Api_Key=your-key
```

You can also skip this and paste the key into the app's sidebar, or use the Pollinations backend, which needs no key.

**Note:** on the free tier, Gemini gives 0 image-generation requests (`429 … limit: 0`). Text works, images don't. To generate with Gemini, enable billing for the key's project in [Google AI Studio](https://aistudio.google.com). Until then, select **Pollinations (free, no key)** in the app. It allows about 1 image per 45 seconds.

## Step 3 — Run the demo

```powershell
streamlit run app.py
```

Your browser opens <http://localhost:8501>. The first start takes about 30 s, because TensorFlow and PyTorch load and Community Forensics downloads its weights once (about 90 MB, cached in `%USERPROFILE%\.cache\huggingface`). Stop the app with **Ctrl+C** in PowerShell.

How to use it:

1. **Sidebar → Backend:** choose *Pollinations (free, no key)*, or *Gemini API* once billing is enabled.
2. **Sidebar → Final verdict from:** *⭐ Community Forensics* (default) or *CIFAKE models (ensemble)*.
3. **✨ Generate from prompt:** type a prompt and click **Generate image**. The image is classified right away. The table shows every model's P(AI); "What a 32x32 CIFAKE model sees" shows how much detail your CIFAKE models lose.
4. **📤 Upload an image:** drop in any photo or AI image. Optionally set the ground truth.
5. **📊 Evaluation log:** click *Add result to evaluation log* under a prediction to collect results, then download them as CSV.

Demo idea: upload a real photo from `evaluation/images/real/`, then generate an AI image of the same subject. Community Forensics separates the two; the CIFAKE models don't.

## Step 4 — Reproduce the evaluation (optional)

All commands run inside the activated `.venv`.

**a) Pretrained detectors + your models, AI images vs real photos** (downloads about 3 GB of weights the first time):

```powershell
python evaluate_hf.py --cifake "C:\Users\asus\Downloads\nandika_project.zip"
```

This writes `evaluation/open_source/leaderboard.md`, plus `report.md` and `per_image.csv` for each model. `--cifake` adds an in-domain check on 200 CIFAKE test images; leave it out if you don't have the zip. To run only some models, add `--models commfor,haywoodsloan`.

**b) Your CIFAKE models only, with test-time augmentation** (also generates images if they are missing):

```powershell
python evaluate.py --backend pollinations --cifake "C:\Users\asus\Downloads\nandika_project.zip"
```

**c) New test images:**

```powershell
python fetch_real_photos.py --per-class 3
```

```powershell
python evaluate.py --backend gemini --n 10
```

```powershell
python evaluate_hf.py --ai-dir evaluation\images\gemini
```

The first command downloads 3 real photos per class. The second generates with Gemini once billing is on. The third evaluates the detectors on those Gemini images.

## Using a detector in your own code

```python
from PIL import Image
import hf_detectors

det = hf_detectors.load_detector("commfor")          # or umm-maybe, organika-sdxl, haywoodsloan, ...
print(det.predict([Image.open("photo.jpg")]))         # -> array([P(AI)])
```

## Next time

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
streamlit run app.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `py -3.12` not found | Install Python 3.12 from python.org and tick "Add to PATH" |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Sidebar: *Community Forensics not available* | `pip install -r requirements.txt` inside `.venv`. The first run needs internet. |
| Gemini `429 … limit: 0` | Free tier has no image quota. Use Pollinations or enable billing. |
| Pollinations `402 FREE_LIMIT` / `503` | Wait about 45 s and try again |
| `ModuleNotFoundError` | You're not in `.venv`. Activate it first. |

## Credits

* Community Forensics: Park & Owens, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors* ([code](https://github.com/JeongsooP/Community-Forensics), MIT licence).
* The other detectors are listed with their Hugging Face repos in `evaluation/open_source/leaderboard.md`.
* Real photos: Wikimedia Commons. Authors and licences are in `evaluation/images/real/CREDITS.csv`.
