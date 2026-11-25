# nyuad — `NYUAD-ComNets/NYUAD_AI-generated_images_detector`

ViT, 3 classes (dalle / sd / real)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.6 s, inference 109.5 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 70% | 60% | 20% | 0.830 |
| JPEG q=75 | 70% | 60% | 20% | 0.830 |
| downscale 50% | 70% | 60% | 20% | 0.830 |
| noise sigma=8 | 70% | 60% | 20% | 0.830 |
| centre crop 70% | 70% | 60% | 20% | 0.810 |

## CIFAKE test sample (32x32 images)

accuracy 50.5% · AUC 0.650 · TPR 1.0% · FPR 0.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 0.787 | 0.735 | 0.753 | 0.767 | 0.530 |
| AI | pollinations/01_automobile.png | 0.997 | 0.997 | 0.997 | 0.998 | 0.941 |
| AI | pollinations/02_bird.png | 0.997 | 0.997 | 0.998 | 0.997 | 0.998 |
| AI | pollinations/03_cat.png | 0.838 | 0.838 | 0.812 | 0.853 | 0.516 |
| AI | pollinations/04_deer.png | 0.004 | 0.004 | 0.004 | 0.004 | 0.024 |
| AI | pollinations/05_dog.png | 0.759 | 0.775 | 0.732 | 0.734 | 0.940 |
| AI | pollinations/06_frog.png | 0.020 | 0.022 | 0.015 | 0.019 | 0.047 |
| AI | pollinations/07_horse.png | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| AI | pollinations/08_ship.png | 0.012 | 0.014 | 0.009 | 0.013 | 0.006 |
| AI | pollinations/09_truck.png | 0.156 | 0.135 | 0.104 | 0.116 | 0.007 |
| real | real/00_airplane.jpg | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 |
| real | real/01_automobile.jpg | 0.974 | 0.972 | 0.976 | 0.972 | 0.978 |
| real | real/02_bird.jpg | 0.002 | 0.002 | 0.002 | 0.003 | 0.024 |
| real | real/03_cat.jpg | 0.002 | 0.003 | 0.002 | 0.003 | 0.006 |
| real | real/04_deer.jpg | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 |
| real | real/05_dog.jpg | 0.018 | 0.018 | 0.015 | 0.017 | 0.011 |
| real | real/06_frog.jpg | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| real | real/07_horse.jpg | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| real | real/08_ship.jpg | 0.009 | 0.009 | 0.008 | 0.008 | 0.003 |
| real | real/09_truck.jpg | 0.986 | 0.987 | 0.987 | 0.987 | 0.653 |
