# Leaderboard — AI vs real

Test set: 10 AI images (`pollinations`) + 10 real photos (`real`). Threshold 0.5. Sorted by accuracy on the original images.

| Model | Accuracy | AI caught (TPR) | Real flagged (FPR) | AUC | Acc: JPEG q=75 | Acc: downscale 50% | Acc: noise sigma=8 | Acc: centre crop 70% | CIFAKE acc | ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **commfor** | 100% | 100% | 0% | 1.000 | 100% | 100% | 100% | 100% | 50% | 108 |
| **haywoodsloan** | 100% | 100% | 0% | 1.000 | 100% | 100% | 90% | 100% | 54% | 416 |
| **ateeqq-siglip** | 100% | 100% | 0% | 1.000 | 100% | 95% | 100% | 100% | 40% | 117 |
| **organika-sdxl** | 95% | 100% | 10% | 1.000 | 95% | 95% | 70% | 90% | 57% | 127 |
| **smogy** | 95% | 90% | 0% | 0.980 | 95% | 95% | 85% | 80% | 51% | 125 |
| **nyuad** | 70% | 60% | 20% | 0.830 | 70% | 70% | 70% | 70% | 50% | 110 |
| **umm-maybe** | 65% | 50% | 20% | 0.620 | 65% | 65% | 65% | 60% | 60% | 129 |
| **XGBoost (handcrafted)** | 60% | 50% | 30% | 0.590 | 60% | 60% | 60% | 45% | 92% | 9 |
| **Random Forest (handcrafted)** | 55% | 50% | 40% | 0.495 | 55% | 55% | 55% | 45% | 88% | 10 |
| **dima806-cifake** | 50% | 100% | 100% | 0.980 | 50% | 50% | 50% | 55% | 98% | 110 |
| **CNN (raw pixels)** | 40% | 10% | 30% | 0.430 | 40% | 40% | 40% | 55% | 94% | 9 |

| Model | Source | Description |
|---|---|---|
| commfor | `OwensLab/commfor-model-384` | Community Forensics ViT-S/16-384 (Park & Owens, CVPR 2025; trained on 4,803 generators) |
| haywoodsloan | `haywoodsloan/ai-image-detector-deploy` | SwinV2, continuously retrained (2024-2026) |
| ateeqq-siglip | `Ateeqq/ai-vs-human-image-detector` | SigLIP fine-tuned on 60k AI + 60k human images |
| organika-sdxl | `Organika/sdxl-detector` | Swin, umm-maybe fine-tuned on SDXL vs Wikimedia photos |
| smogy | `Smogy/SMOGY-Ai-images-detector` | Swin fine-tuned on 50k+ images |
| nyuad | `NYUAD-ComNets/NYUAD_AI-generated_images_detector` | ViT, 3 classes (dalle / sd / real) |
| umm-maybe | `umm-maybe/AI-image-detector` | Swin, trained mostly on AI art (2022) |
| XGBoost (handcrafted) | `models/xgboost.json` | your CIFAKE baseline (32x32 input) |
| Random Forest (handcrafted) | `models/random_forest.joblib` | your CIFAKE baseline (32x32 input) |
| dima806-cifake | `dima806/ai_vs_real_image_detection` | ViT fine-tuned on CIFAKE (same dataset as your models) |
| CNN (raw pixels) | `models/cnn.keras` | your CIFAKE baseline (32x32 input) |
