# CPE 342 Machine Learning: Karena Player Intelligence (Kaggle project)

Five ML tasks on synthetic Karena player data. Each task folder holds the notebooks that produce its column of `submission.csv` (data audit → EDA → models → predictions + insights).

**Final submission: public leaderboard 0.87016** (weighted over the 5 tasks). The answer files are not stored in this repository; every column of `submission.csv` is produced by running the notebooks below.

| Folder | Task | Metric | Notebook(s) | Out-of-fold result* |
|---|---|---|---|---|
| `task1` | Anti-cheat (cheater / not) | F2 | `task1_anticheat_pipeline.ipynb` | F2 0.8376 (AUC 0.893, threshold 0.175) |
| `task2` | Player segment (4 classes) | macro-F1 | `task2_segmentation_pipeline.ipynb` | macro-F1 0.7570 |
| `task3` | 30-day spending (THB) | NMAE | `task3_spending_pipeline.ipynb` | NMAE score 0.7939 (MAE 2,726) |
| `task4` | Game title from a screenshot (5 classes) | macro-F1 | `task4_dino7b_lora_seed7.ipynb`, `task4_dino7b_lora_seed8.ipynb` → `task4_final_blend.ipynb` | macro-F1 0.9975 (clean OOF) |
| `task5` | Account anomaly (unsupervised) | F3 | `task5_anomaly_pipeline.ipynb` | no labels, not measurable locally |

\*5-fold cross-validation on the provided training sets.

## Repository layout
```
task1/ … task5/    final code: the notebooks that produce submission.csv (one folder per task)
requirements.txt   Python packages
```

## Approach per task
1. **Task 1:** LightGBM + HistGradientBoosting + CatBoost (GPU), 5 folds × 3 seeds, blended in logit space with weights fitted on the out-of-fold AUC. Every feature has about 20–24% missing values; training folds get an extra copy with a fixed 10% per-column NaN mask (chosen on train-only CV; imputation was tested and lost). The F2 threshold (0.175) is tuned on the out-of-fold predictions. Hyperparameters: Optuna on train folds.
2. **Task 2:** LightGBM, CatBoost and a KDE naive Bayes, stacked by a logistic regression on their out-of-fold log-probabilities, then per-class probability weights tuned on the out-of-fold macro-F1. `random_metric_*` columns are dropped (pure noise per the datacard).
3. **Task 3:** NMAE rewards the conditional median. Missing values are repaired with exact row identities (`historical_spending = total_transactions × avg_transaction_value`, …) and a classifier for `segment`; then a hurdle model (LightGBM + CatBoost): P(spend > 0) and, for spenders, the log amount relative to `historical_spending` (L1), read out as the mixture median (exactly 0 when P ≤ 0.5). σ of the readout is tuned on out-of-fold predictions.
4. **Task 4:** DINOv3 ViT-7B/16 (self-supervised pre-training, the only pretrained model in the project) fine-tuned with LoRA (rank 16, all linear layers) on a B200 GPU: 192×384 input, damage augmentation + Mixup/CutMix, EMA weights, 15 epochs, flip TTA. Two training seeds; each = 5 fold models + one full-data model (test = ½ fold mean + ½ full-data model). The answer is the mean of the two seeds with plain argmax (the class-prior rule was tested on out-of-fold predictions and not used). Labels follow `datacard.txt` (0 = 7-11 Knights, 1 = FiveN, 2 = Free Fried, 3 = PUBE, 4 = Roblock).
5. **Task 5:** a planted behaviour leaves an empty gap between itself and the crowd. For every column the gaps in each tail are scored by how many accounts the dense side would have put inside them; columns that isolate the same accounts are linked into six behaviour groups (1,812 accounts). One more group is odd only in combination: high spread of KDA and of accuracy inside a narrow band of `performance_stability` (322 accounts, cut-off from an expected-F3 rule). All 2,134 accounts are flagged.

## Rules followed
- Provided datasets only. No test labels, no pseudo-labelling, no AutoML (Optuna for hyperparameters only). Pretrained weights only in Task 4.
- **Every parameter is computed without looking at the test set** (TA ruling: priors, thresholds, scalers and design choices must come from train/validation only). Tasks 1–4 never measure test rows: test is only predicted. Task 5 has no training set by design (only `test.csv`, no labels), so its unsupervised method necessarily works on the test rows.

## Submission format
- Columns `id, task1 … task5`; labels are **integers** (task1, 2, 4, 5), task3 is a float. Writing task2 as `3.0` instead of `3` scores far lower, so keep integers.
- Tasks 1, 2, 3, 5: row `i` of the submission is row `i` of that task's `test.csv` (`ANS00001` = first row). **Task 4** is matched by `ANS…` id: its `test.csv` (columns `id`, `file_name`, `label`) maps each id to an image, and its row order differs from the other tasks.

## How to run
- **Tasks 1–3, 5:** put `train.csv`, `test.csv` next to each notebook and run it top to bottom. Each notebook writes its column into `../submission.csv` (started from `../sample_submission.csv` if missing) and a per-task file `submission_taskN.csv`. CatBoost uses a GPU when one is available.
- **Task 4 training** (`task4/`): needs a large GPU (run on a 4× B200 machine: about 2 h per fold or full-data model on one GPU at 192×384, about 6 h for two seeds) and the gated DINOv3 weights (accept the licence on HuggingFace and `export HF_TOKEN=…` in the terminal; the token is never stored in a notebook). `run_lora_4gpu.sh` spreads the folds of each seed over the GPUs of one machine (`TRAIN_SEEDS="7 8" bash run_lora_4gpu.sh`), using `run_cells.py` for the parallel workers. The images come from the Google Drive link in `task4/README.md`.
- **Task 4 answers:** `task4_final_blend.ipynb` runs on CPU in about a minute from the saved probabilities in `task4/results/` (`*_seed7`, `*_seed8`) and writes the `task4` column.
- Python packages: `pip install -r requirements.txt` (the torch / timm / transformers / peft / gdown lines are needed only for Task 4 training).

## Known limits
- Task 5 has no labels. The behaviour groups are clearly separate crowds in the data, but the four security flags confirm only the travel / location group; the skill-jump group could be players who genuinely improved.
- Task 4 augmentation: a train-only check supports most of the strengths (brightness, saturation and colour cast stay inside train's own range for 89–96% of augmented images), while the augmented images are smoother and get black corners more often than train.
- Out-of-fold scores come from random stratified folds; near-duplicate frames of the same stream can sit on both sides of a split, so the Task 4 out-of-fold score (0.997) is likely optimistic.
