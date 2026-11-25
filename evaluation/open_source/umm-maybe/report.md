# umm-maybe — `umm-maybe/AI-image-detector`

Swin, trained mostly on AI art (2022)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 3.4 s, inference 129.0 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 65% | 50% | 20% | 0.620 |
| JPEG q=75 | 65% | 50% | 20% | 0.710 |
| downscale 50% | 65% | 50% | 20% | 0.670 |
| noise sigma=8 | 65% | 80% | 50% | 0.730 |
| centre crop 70% | 60% | 50% | 30% | 0.700 |

## CIFAKE test sample (32x32 images)

accuracy 59.5% · AUC 0.595 · TPR 44.0% · FPR 25.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 0.289 | 0.364 | 0.407 | 0.614 | 0.204 |
| AI | pollinations/01_automobile.png | 0.684 | 0.736 | 0.748 | 0.933 | 0.840 |
| AI | pollinations/02_bird.png | 0.100 | 0.120 | 0.116 | 0.281 | 0.083 |
| AI | pollinations/03_cat.png | 0.053 | 0.063 | 0.084 | 0.135 | 0.072 |
| AI | pollinations/04_deer.png | 0.162 | 0.170 | 0.225 | 0.591 | 0.111 |
| AI | pollinations/05_dog.png | 0.831 | 0.834 | 0.836 | 0.906 | 0.872 |
| AI | pollinations/06_frog.png | 0.828 | 0.794 | 0.883 | 0.896 | 0.894 |
| AI | pollinations/07_horse.png | 0.669 | 0.745 | 0.717 | 0.893 | 0.800 |
| AI | pollinations/08_ship.png | 0.279 | 0.364 | 0.374 | 0.715 | 0.417 |
| AI | pollinations/09_truck.png | 0.573 | 0.575 | 0.620 | 0.775 | 0.659 |
| real | real/00_airplane.jpg | 0.472 | 0.344 | 0.480 | 0.762 | 0.421 |
| real | real/01_automobile.jpg | 0.914 | 0.870 | 0.917 | 0.870 | 0.809 |
| real | real/02_bird.jpg | 0.328 | 0.232 | 0.312 | 0.326 | 0.196 |
| real | real/03_cat.jpg | 0.033 | 0.022 | 0.037 | 0.075 | 0.005 |
| real | real/04_deer.jpg | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 |
| real | real/05_dog.jpg | 0.083 | 0.046 | 0.084 | 0.224 | 0.055 |
| real | real/06_frog.jpg | 0.611 | 0.591 | 0.656 | 0.751 | 0.665 |
| real | real/07_horse.jpg | 0.078 | 0.065 | 0.079 | 0.133 | 0.051 |
| real | real/08_ship.jpg | 0.355 | 0.314 | 0.385 | 0.685 | 0.211 |
| real | real/09_truck.jpg | 0.483 | 0.343 | 0.394 | 0.673 | 0.659 |
