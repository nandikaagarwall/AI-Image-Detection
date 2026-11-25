# Random Forest (handcrafted) — `models/random_forest.joblib`

your CIFAKE baseline (32x32 input)

* Test set: 10 AI images (`pollinations`) + 10 real photos (`real`)
* Threshold 0.5. acc = accuracy, TPR = AI images detected, FPR = real photos wrongly flagged as AI
* Load time 0.0 s, inference 9.5 ms/image on CPU

## AI vs real, per perturbation

| Condition | Accuracy | TPR (AI caught) | FPR (real flagged) | AUC |
|---|---|---|---|---|
| original | 55% | 50% | 40% | 0.495 |
| JPEG q=75 | 55% | 50% | 40% | 0.490 |
| downscale 50% | 55% | 50% | 40% | 0.490 |
| noise sigma=8 | 55% | 50% | 40% | 0.520 |
| centre crop 70% | 45% | 40% | 50% | 0.480 |

## CIFAKE test sample (32x32 images)

accuracy 87.5% · AUC 0.954 · TPR 83.0% · FPR 8.0%

## P(AI) per image

| Truth | Image | original | JPEG q=75 | downscale 50% | noise sigma=8 | centre crop 70% |
|---|---|---|---|---|---|---|
| AI | pollinations/00_airplane.png | 0.640 | 0.657 | 0.640 | 0.617 | 0.590 |
| AI | pollinations/01_automobile.png | 0.653 | 0.663 | 0.647 | 0.633 | 0.520 |
| AI | pollinations/02_bird.png | 0.417 | 0.387 | 0.393 | 0.383 | 0.370 |
| AI | pollinations/03_cat.png | 0.530 | 0.520 | 0.527 | 0.540 | 0.353 |
| AI | pollinations/04_deer.png | 0.197 | 0.167 | 0.197 | 0.237 | 0.213 |
| AI | pollinations/05_dog.png | 0.380 | 0.357 | 0.367 | 0.367 | 0.193 |
| AI | pollinations/06_frog.png | 0.403 | 0.433 | 0.443 | 0.437 | 0.417 |
| AI | pollinations/07_horse.png | 0.653 | 0.637 | 0.630 | 0.637 | 0.607 |
| AI | pollinations/08_ship.png | 0.570 | 0.590 | 0.570 | 0.573 | 0.717 |
| AI | pollinations/09_truck.png | 0.380 | 0.363 | 0.380 | 0.337 | 0.403 |
| real | real/00_airplane.jpg | 0.417 | 0.400 | 0.390 | 0.347 | 0.293 |
| real | real/01_automobile.jpg | 0.780 | 0.767 | 0.763 | 0.767 | 0.750 |
| real | real/02_bird.jpg | 0.597 | 0.577 | 0.617 | 0.613 | 0.537 |
| real | real/03_cat.jpg | 0.283 | 0.277 | 0.260 | 0.293 | 0.290 |
| real | real/04_deer.jpg | 0.287 | 0.280 | 0.320 | 0.323 | 0.400 |
| real | real/05_dog.jpg | 0.377 | 0.403 | 0.397 | 0.337 | 0.350 |
| real | real/06_frog.jpg | 0.700 | 0.673 | 0.697 | 0.687 | 0.710 |
| real | real/07_horse.jpg | 0.333 | 0.333 | 0.350 | 0.340 | 0.310 |
| real | real/08_ship.jpg | 0.423 | 0.417 | 0.423 | 0.427 | 0.543 |
| real | real/09_truck.jpg | 0.657 | 0.670 | 0.657 | 0.637 | 0.547 |
