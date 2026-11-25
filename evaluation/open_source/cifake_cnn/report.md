# CNN (raw pixels) — `models/cnn.keras`

your CIFAKE baseline (32x32 input)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.0 s, inference 9.2 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 40% | 10% | 30% | 0.430 |
| JPEG q=75 | 40% | 10% | 30% | 0.430 |
| downscale 50% | 40% | 10% | 30% | 0.430 |
| noise sigma=8 | 40% | 10% | 30% | 0.420 |
| centre crop 70% | 55% | 20% | 10% | 0.470 |

## CIFAKE test sample (32x32 images)

accuracy 93.5% · AUC 0.988 · TPR 88.0% · FPR 1.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 0.002 | 0.002 | 0.002 | 0.003 | 0.000 |
| AI | pollinations/01_automobile.png | 0.947 | 0.951 | 0.947 | 0.940 | 0.756 |
| AI | pollinations/02_bird.png | 0.187 | 0.180 | 0.164 | 0.170 | 0.259 |
| AI | pollinations/03_cat.png | 0.015 | 0.014 | 0.013 | 0.014 | 0.001 |
| AI | pollinations/04_deer.png | 0.001 | 0.001 | 0.001 | 0.001 | 0.025 |
| AI | pollinations/05_dog.png | 0.088 | 0.086 | 0.079 | 0.088 | 0.006 |
| AI | pollinations/06_frog.png | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 |
| AI | pollinations/07_horse.png | 0.203 | 0.214 | 0.169 | 0.199 | 0.705 |
| AI | pollinations/08_ship.png | 0.316 | 0.302 | 0.297 | 0.286 | 0.270 |
| AI | pollinations/09_truck.png | 0.002 | 0.002 | 0.002 | 0.002 | 0.001 |
| real | real/00_airplane.jpg | 0.008 | 0.008 | 0.007 | 0.008 | 0.085 |
| real | real/01_automobile.jpg | 0.938 | 0.939 | 0.938 | 0.931 | 0.989 |
| real | real/02_bird.jpg | 0.063 | 0.062 | 0.059 | 0.055 | 0.060 |
| real | real/03_cat.jpg | 0.035 | 0.034 | 0.032 | 0.039 | 0.016 |
| real | real/04_deer.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/05_dog.jpg | 0.033 | 0.032 | 0.033 | 0.035 | 0.038 |
| real | real/06_frog.jpg | 0.031 | 0.025 | 0.030 | 0.028 | 0.115 |
| real | real/07_horse.jpg | 0.169 | 0.171 | 0.151 | 0.183 | 0.099 |
| real | real/08_ship.jpg | 0.679 | 0.675 | 0.667 | 0.655 | 0.001 |
| real | real/09_truck.jpg | 0.935 | 0.923 | 0.931 | 0.925 | 0.017 |
