# haywoodsloan — `haywoodsloan/ai-image-detector-deploy`

SwinV2, continuously retrained (2024-2026)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 1.5 s, inference 416.1 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 100% | 100% | 0% | 1.000 |
| JPEG q=75 | 100% | 100% | 0% | 1.000 |
| downscale 50% | 100% | 100% | 0% | 1.000 |
| noise sigma=8 | 90% | 90% | 10% | 0.970 |
| centre crop 70% | 100% | 100% | 0% | 1.000 |

## CIFAKE test sample (32x32 images)

accuracy 54.0% · AUC 0.586 · TPR 10.0% · FPR 2.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 0.959 | 0.928 | 0.954 | 0.815 | 0.957 |
| AI | pollinations/01_automobile.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/02_bird.png | 0.994 | 0.994 | 0.992 | 0.995 | 0.992 |
| AI | pollinations/03_cat.png | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| AI | pollinations/04_deer.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/05_dog.png | 1.000 | 1.000 | 1.000 | 0.998 | 1.000 |
| AI | pollinations/06_frog.png | 0.996 | 0.989 | 0.981 | 0.196 | 0.994 |
| AI | pollinations/07_horse.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/08_ship.png | 0.996 | 0.994 | 0.993 | 0.931 | 0.987 |
| AI | pollinations/09_truck.png | 0.999 | 0.999 | 0.999 | 0.997 | 0.974 |
| real | real/00_airplane.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| real | real/01_automobile.jpg | 0.001 | 0.000 | 0.001 | 0.002 | 0.001 |
| real | real/02_bird.jpg | 0.000 | 0.001 | 0.001 | 0.002 | 0.000 |
| real | real/03_cat.jpg | 0.000 | 0.000 | 0.001 | 0.002 | 0.001 |
| real | real/04_deer.jpg | 0.064 | 0.020 | 0.060 | 0.947 | 0.005 |
| real | real/05_dog.jpg | 0.000 | 0.001 | 0.000 | 0.001 | 0.000 |
| real | real/06_frog.jpg | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 |
| real | real/07_horse.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| real | real/08_ship.jpg | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 |
| real | real/09_truck.jpg | 0.011 | 0.003 | 0.014 | 0.009 | 0.002 |
