# dima806-cifake — `dima806/ai_vs_real_image_detection`

ViT fine-tuned on CIFAKE (same dataset as your models)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.8 s, inference 110.2 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 50% | 100% | 100% | 0.980 |
| JPEG q=75 | 50% | 100% | 100% | 0.980 |
| downscale 50% | 50% | 100% | 100% | 0.980 |
| noise sigma=8 | 50% | 100% | 100% | 0.980 |
| centre crop 70% | 55% | 100% | 90% | 1.000 |

## CIFAKE test sample (32x32 images)

accuracy 98.0% · AUC 0.997 · TPR 98.0% · FPR 2.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 0.998 | 0.998 | 0.998 | 0.998 | 0.996 |
| AI | pollinations/01_automobile.png | 0.997 | 0.997 | 0.997 | 0.997 | 0.991 |
| AI | pollinations/02_bird.png | 0.990 | 0.990 | 0.990 | 0.989 | 0.984 |
| AI | pollinations/03_cat.png | 0.876 | 0.887 | 0.883 | 0.895 | 0.873 |
| AI | pollinations/04_deer.png | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |
| AI | pollinations/05_dog.png | 0.988 | 0.988 | 0.990 | 0.988 | 0.973 |
| AI | pollinations/06_frog.png | 0.995 | 0.995 | 0.996 | 0.995 | 0.987 |
| AI | pollinations/07_horse.png | 0.996 | 0.996 | 0.997 | 0.996 | 0.998 |
| AI | pollinations/08_ship.png | 0.993 | 0.992 | 0.993 | 0.991 | 0.989 |
| AI | pollinations/09_truck.png | 0.991 | 0.991 | 0.991 | 0.993 | 0.981 |
| real | real/00_airplane.jpg | 0.796 | 0.799 | 0.822 | 0.792 | 0.804 |
| real | real/01_automobile.jpg | 0.548 | 0.547 | 0.551 | 0.528 | 0.864 |
| real | real/02_bird.jpg | 0.537 | 0.538 | 0.535 | 0.537 | 0.450 |
| real | real/03_cat.jpg | 0.970 | 0.964 | 0.972 | 0.966 | 0.792 |
| real | real/04_deer.jpg | 0.688 | 0.668 | 0.694 | 0.709 | 0.579 |
| real | real/05_dog.jpg | 0.579 | 0.570 | 0.574 | 0.581 | 0.617 |
| real | real/06_frog.jpg | 0.859 | 0.859 | 0.863 | 0.857 | 0.819 |
| real | real/07_horse.jpg | 0.663 | 0.651 | 0.656 | 0.655 | 0.668 |
| real | real/08_ship.jpg | 0.943 | 0.953 | 0.937 | 0.962 | 0.867 |
| real | real/09_truck.jpg | 0.660 | 0.661 | 0.659 | 0.660 | 0.643 |
