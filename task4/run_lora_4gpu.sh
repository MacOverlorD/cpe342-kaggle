#!/bin/bash
# DINOv3-7B + LoRA (5 folds + full-data model, one set per training seed) spread over all GPUs of this machine.
# Needs in this folder: this script, run_cells.py, task4_dino7b_lora_seed7.ipynb. Then:
#   export HF_TOKEN=hf_...        # DINOv3 weights are gated: accept the licence on HuggingFace first
#   bash run_lora_4gpu.sh 2>&1 | tee run.log
# Data is downloaded from the competition's Google Drive link unless train.csv etc. are already here.
# Safe to start again after a crash: finished runs are reused, not retrained.
set -u
cd "$(dirname "$0")"
HERE=$(pwd)
NB=${NB:-task4_dino7b_lora_seed7}   # the training notebook (TRAIN_SEED sets the seed)
DRIVE_ID=1C8csylbBo3Oab24ODGKPVom3_SDUJ_Qw   # from the competition's 04_image_classification/README.md
RUNS=(0 1 2 3 4 full)

# 1) data: find train.csv/test.csv/train/test, downloading them if missing
find_data() { dirname "$(find "$HERE" -name train.csv -not -path '*/run_*' -not -path '*/results_*' 2>/dev/null | head -n 1)"; }
D=$(find_data)
if [ ! -f "$D/train.csv" ]; then
  echo "$(date '+%F %T') downloading data"
  gdown "https://drive.google.com/uc?id=$DRIVE_ID" -O data.zip || echo "download failed: upload the data zip here as data.zip and run again"
  [ -f data.zip ] && mkdir -p data && python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('data')" && rm data.zip
  D=$(find_data)
fi
for p in train.csv test.csv train test; do [ -e "$D/$p" ] || { echo "missing $D/$p"; exit 1; }; done
echo "data: $D ($(ls "$D/train" | wc -l) train images, $(ls "$D/test" | wc -l) test images)"

DIR="$HERE/run_$NB"; mkdir -p "$DIR"; cp "$HERE/$NB.ipynb" "$HERE/run_cells.py" "$DIR/"
for p in train.csv test.csv train test; do ln -sfn "$D/$p" "$DIR/$p"; done
cd "$DIR"

# 2) parallel workers: every (training seed, run) job is dealt round-robin over the GPUs.
#    Default: training seed 7 on the seed-42 folds, 15 epochs -> 6 jobs (4 GPUs: 0+4 | 1+full | 2 | 3).
#    Override with e.g.  TRAIN_SEEDS="7" EPOCHS=10 bash run_lora_4gpu.sh   (TRAIN_SEEDS=42 = the original run)
#    A seed can have its own image size (height,width):  TRAIN_SEEDS="8 9/224,448"
SEEDS=(${TRAIN_SEEDS:-7}); export EPOCHS=${EPOCHS:-15}
JOBS=(); for s in "${SEEDS[@]}"; do for r in "${RUNS[@]}"; do JOBS+=("$s:$r"); done; done
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
LANES=$(( NG < ${#JOBS[@]} ? NG : ${#JOBS[@]} ))
echo "$(date '+%F %T') training seeds ${SEEDS[*]}, $EPOCHS epochs, ${#JOBS[@]} jobs on $LANES GPU(s)"
if [ "$LANES" -gt 1 ]; then
  for ((k = 0; k < LANES; k++)); do
    plan=""; for s in "${SEEDS[@]}"; do          # this GPU's runs, grouped by seed: "7=0,4 8=2"
      mine=""; for ((j = k; j < ${#JOBS[@]}; j += LANES)); do [ "${JOBS[$j]%%:*}" = "$s" ] && mine="$mine${mine:+,}${JOBS[$j]#*:}"; done
      [ -n "$mine" ] && plan="$plan $s=$mine"
    done
    echo "$(date '+%F %T') GPU $k:$plan (log: run_$NB/worker_gpu$k.log)"
    ( ok=1; for sr in $plan; do
        spec=${sr%%=*}; size=192,384; [[ $spec == */* ]] && size=${spec#*/}
        env NOT_KAGGLE=1 CUDA_VISIBLE_DEVICES=$k TRAIN_SEED=${spec%%/*} SIZE=$size ONLY_RUNS=${sr#*=} python run_cells.py "$NB.ipynb" >> "worker_gpu$k.log" 2>&1 || ok=0
      done
      [ $ok = 1 ] && echo "$(date '+%F %T') GPU $k done" || echo "$(date '+%F %T') GPU $k FAILED (see run_$NB/worker_gpu$k.log)" ) &
  done
  wait
fi

# 3) final pass per seed on GPU 0: reuses every finished run (trains any that failed above), then evaluates and writes the outputs
for spec in "${SEEDS[@]}"; do
  s=${spec%%/*}; size=192,384; [[ $spec == */* ]] && size=${spec#*/}
  echo "$(date '+%F %T') final pass, training seed $s, size $size"
  env NOT_KAGGLE=1 CUDA_VISIBLE_DEVICES=0 TRAIN_SEED=$s SIZE=$size jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1     --output "$NB-run_t$s.ipynb" "$NB.ipynb" > "final_t$s.log" 2>&1 && echo "$(date '+%F %T') final pass seed $s done"     || echo "$(date '+%F %T') final pass seed $s FAILED (see run_$NB/final_t$s.log)"
done

RES="$HERE/results_$NB"; mkdir -p "$RES"
cp task4_oof_*.npy task4_test_probs_*.npy "$NB"-run*.ipynb worker_gpu*.log "$RES/" 2>/dev/null
ls -la "$RES"
