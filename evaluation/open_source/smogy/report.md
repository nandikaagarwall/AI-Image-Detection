# smogy — `Smogy/SMOGY-Ai-images-detector`

Swin fine-tuned on 50k+ images

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.9 s, inference 124.9 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 95% | 90% | 0% | 0.980 |
| JPEG q=75 | 95% | 90% | 0% | 1.000 |
| downscale 50% | 95% | 90% | 0% | 0.980 |
| noise sigma=8 | 85% | 90% | 20% | 0.980 |
| centre crop 70% | 80% | 90% | 30% | 0.960 |

## CIFAKE test sample (32x32 images)

accuracy 51.0% · AUC 0.743 · TPR 99.0% · FPR 97.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/01_automobile.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/02_bird.png | 0.080 | 0.394 | 0.027 | 0.015 | 0.047 |
| AI | pollinations/03_cat.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/04_deer.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/05_dog.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/06_frog.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/07_horse.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/08_ship.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/09_truck.png | 1.000 | 1.000 | 0.997 | 0.998 | 1.000 |
| real | real/00_airplane.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.982 |
| real | real/01_automobile.jpg | 0.285 | 0.037 | 0.073 | 0.912 | 1.000 |
| real | real/02_bird.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/03_cat.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/04_deer.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.077 |
| real | real/05_dog.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/06_frog.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/07_horse.jpg | 0.000 | 0.000 | 0.000 | 0.006 | 0.998 |
| real | real/08_ship.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/09_truck.jpg | 0.093 | 0.062 | 0.073 | 0.600 | 0.000 |
