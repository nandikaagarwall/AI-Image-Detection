# organika-sdxl — `Organika/sdxl-detector`

Swin, umm-maybe fine-tuned on SDXL vs Wikimedia photos

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.9 s, inference 126.6 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 95% | 100% | 10% | 1.000 |
| JPEG q=75 | 95% | 100% | 10% | 0.990 |
| downscale 50% | 95% | 100% | 10% | 1.000 |
| noise sigma=8 | 70% | 40% | 0% | 0.960 |
| centre crop 70% | 90% | 100% | 20% | 0.960 |

## CIFAKE test sample (32x32 images)

accuracy 57.5% · AUC 0.640 · TPR 25.0% · FPR 10.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 1.000 | 1.000 | 1.000 | 0.009 | 1.000 |
| AI | pollinations/01_automobile.png | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| AI | pollinations/02_bird.png | 1.000 | 1.000 | 1.000 | 0.486 | 1.000 |
| AI | pollinations/03_cat.png | 1.000 | 1.000 | 1.000 | 0.011 | 1.000 |
| AI | pollinations/04_deer.png | 1.000 | 1.000 | 1.000 | 0.992 | 1.000 |
| AI | pollinations/05_dog.png | 1.000 | 1.000 | 1.000 | 0.997 | 0.998 |
| AI | pollinations/06_frog.png | 0.999 | 0.999 | 0.999 | 0.814 | 0.998 |
| AI | pollinations/07_horse.png | 1.000 | 1.000 | 1.000 | 0.442 | 1.000 |
| AI | pollinations/08_ship.png | 1.000 | 0.908 | 0.999 | 0.000 | 0.992 |
| AI | pollinations/09_truck.png | 1.000 | 0.999 | 0.999 | 0.002 | 0.995 |
| real | real/00_airplane.jpg | 0.002 | 0.037 | 0.014 | 0.000 | 0.604 |
| real | real/01_automobile.jpg | 0.996 | 0.995 | 0.996 | 0.173 | 1.000 |
| real | real/02_bird.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/03_cat.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/04_deer.jpg | 0.000 | 0.000 | 0.009 | 0.000 | 0.000 |
| real | real/05_dog.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/06_frog.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/07_horse.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| real | real/08_ship.jpg | 0.000 | 0.049 | 0.001 | 0.000 | 0.000 |
| real | real/09_truck.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
