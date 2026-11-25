# XGBoost (handcrafted) — `models/xgboost.json`

your CIFAKE baseline (32x32 input)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.0 s, inference 9.0 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 60% | 50% | 30% | 0.590 |
| JPEG q=75 | 60% | 50% | 30% | 0.570 |
| downscale 50% | 60% | 50% | 30% | 0.560 |
| noise sigma=8 | 60% | 50% | 30% | 0.590 |
| centre crop 70% | 45% | 20% | 30% | 0.520 |

## CIFAKE test sample (32x32 images)

accuracy 92.5% · AUC 0.978 · TPR 91.0% · FPR 6.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 0.680 | 0.733 | 0.684 | 0.756 | 0.469 |
| AI | pollinations/01_automobile.png | 0.790 | 0.871 | 0.784 | 0.820 | 0.412 |
| AI | pollinations/02_bird.png | 0.274 | 0.271 | 0.244 | 0.158 | 0.488 |
| AI | pollinations/03_cat.png | 0.671 | 0.612 | 0.659 | 0.641 | 0.082 |
| AI | pollinations/04_deer.png | 0.001 | 0.001 | 0.003 | 0.003 | 0.002 |
| AI | pollinations/05_dog.png | 0.312 | 0.256 | 0.201 | 0.202 | 0.015 |
| AI | pollinations/06_frog.png | 0.137 | 0.142 | 0.162 | 0.114 | 0.033 |
| AI | pollinations/07_horse.png | 0.932 | 0.907 | 0.911 | 0.898 | 0.909 |
| AI | pollinations/08_ship.png | 0.678 | 0.691 | 0.637 | 0.518 | 0.928 |
| AI | pollinations/09_truck.png | 0.179 | 0.195 | 0.147 | 0.108 | 0.152 |
| real | real/00_airplane.jpg | 0.231 | 0.306 | 0.214 | 0.111 | 0.096 |
| real | real/01_automobile.jpg | 0.993 | 0.989 | 0.991 | 0.994 | 0.974 |
| real | real/02_bird.jpg | 0.769 | 0.735 | 0.726 | 0.713 | 0.231 |
| real | real/03_cat.jpg | 0.031 | 0.010 | 0.022 | 0.021 | 0.004 |
| real | real/04_deer.jpg | 0.011 | 0.010 | 0.015 | 0.011 | 0.012 |
| real | real/05_dog.jpg | 0.218 | 0.230 | 0.329 | 0.182 | 0.101 |
| real | real/06_frog.jpg | 0.297 | 0.304 | 0.220 | 0.329 | 0.754 |
| real | real/07_horse.jpg | 0.004 | 0.009 | 0.008 | 0.007 | 0.029 |
| real | real/08_ship.jpg | 0.061 | 0.054 | 0.042 | 0.037 | 0.553 |
| real | real/09_truck.jpg | 0.870 | 0.855 | 0.855 | 0.862 | 0.304 |
