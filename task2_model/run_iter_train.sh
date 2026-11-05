#!/bin/bash
# Очередь обучения для итераций (ITERATIONS.md): wxs1, wxs2 - та же модель wx с другим seed (ансамбль, итерация 2);
# wxphys - U-Net с физическим каналом (итерация 4, ждёт out_v3/phys_T_train_era5.done со всеми кусками).
# Возобновляемо: 07_train_v3 продолжает с последней эпохи (<run>_last.pt), готовая модель - <run>_done.
cd "$(dirname "$0")"; PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"
tr() { local run=$1; shift; [ -f out_v3/${run}_done ] && { echo "skip $run"; return 0; }
  echo "=== train $run $(date +%H:%M)"; env RUN=$run "$@" $PY scripts/07_train_v3.py >> logs/07_train_$run.txt 2>&1 && echo "ok $run $(date +%H:%M)" || { echo "FAIL $run"; exit 1; }; }
for s in "$@"; do case $s in
  seeds) tr wxs1 SEED=1; tr wxs2 SEED=2 ;;
  phys)  while [ "$(wc -l < out_v3/phys_T_train_era5.done 2>/dev/null || echo 0)" -lt 30 ]; do sleep 60; done
         tr wxphys PHYS=1 SEED=0 ;;
esac; done
echo "train queue done $(date +%H:%M)"
