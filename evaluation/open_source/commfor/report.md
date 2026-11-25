# commfor — `OwensLab/commfor-model-384`

Community Forensics ViT-S/16-384 (Park & Owens, CVPR 2025; trained on 4,803 generators)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 3.1 s, inference 108.5 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 100% | 100% | 0% | 1.000 |
| JPEG q=75 | 100% | 100% | 0% | 1.000 |
| downscale 50% | 100% | 100% | 0% | 1.000 |
| noise sigma=8 | 100% | 100% | 0% | 1.000 |
| centre crop 70% | 100% | 100% | 0% | 1.000 |

## CIFAKE test sample (32x32 images)

accuracy 50.0% · AUC 0.404 · TPR 0.0% · FPR 0.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 1.000 | 0.999 | 1.000 | 0.670 | 0.995 |
| AI | pollinations/01_automobile.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/02_bird.png | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| AI | pollinations/03_cat.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/04_deer.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/05_dog.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/06_frog.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/07_horse.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/08_ship.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/09_truck.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| real | real/00_airplane.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/01_automobile.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.176 |
| real | real/02_bird.jpg | 0.000 | 0.001 | 0.000 | 0.003 | 0.000 |
| real | real/03_cat.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/04_deer.jpg | 0.008 | 0.003 | 0.013 | 0.014 | 0.003 |
| real | real/05_dog.jpg | 0.002 | 0.001 | 0.002 | 0.000 | 0.000 |
| real | real/06_frog.jpg | 0.002 | 0.001 | 0.001 | 0.000 | 0.001 |
| real | real/07_horse.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/08_ship.jpg | 0.010 | 0.002 | 0.010 | 0.000 | 0.000 |
| real | real/09_truck.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
