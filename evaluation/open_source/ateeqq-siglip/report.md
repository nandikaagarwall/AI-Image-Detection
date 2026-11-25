# ateeqq-siglip — `Ateeqq/ai-vs-human-image-detector`

SigLIP fine-tuned on 60k AI + 60k human images

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.8 s, inference 117.2 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 100% | 100% | 0% | 1.000 |
| JPEG q=75 | 100% | 100% | 0% | 1.000 |
| downscale 50% | 95% | 100% | 10% | 1.000 |
| noise sigma=8 | 100% | 100% | 0% | 1.000 |
| centre crop 70% | 100% | 100% | 0% | 1.000 |

## CIFAKE test sample (32x32 images)

accuracy 40.0% · AUC 0.367 · TPR 49.0% · FPR 69.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| AI | pollinations/01_automobile.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/02_bird.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/03_cat.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/04_deer.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/05_dog.png | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| AI | pollinations/06_frog.png | 1.000 | 0.999 | 0.999 | 0.999 | 0.996 |
| AI | pollinations/07_horse.png | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AI | pollinations/08_ship.png | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| AI | pollinations/09_truck.png | 1.000 | 1.000 | 1.000 | 1.000 | 0.525 |
| real | real/00_airplane.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/01_automobile.jpg | 0.017 | 0.015 | 0.020 | 0.008 | 0.005 |
| real | real/02_bird.jpg | 0.043 | 0.156 | 0.587 | 0.054 | 0.000 |
| real | real/03_cat.jpg | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 |
| real | real/04_deer.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/05_dog.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/06_frog.jpg | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| real | real/07_horse.jpg | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| real | real/08_ship.jpg | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| real | real/09_truck.jpg | 0.008 | 0.008 | 0.014 | 0.006 | 0.002 |
