#!/bin/bash
# Итерации улучшения модели (ITERATIONS.md). Возобновляемо: выполненный шаг отмечается файлом logs/iter_done/<шаг>.
# Запуск: bash run_iter.sh [шаги...]; прогресс - logs/run_iter.txt
cd "$(dirname "$0")"; PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"; mkdir -p logs/iter_done
step() { local id=$1; shift; [ -f logs/iter_done/$id ] && { echo "skip $id"; return 0; }
  echo "=== $id $(date +%H:%M) : $*"; env "$@" > logs/iter_$id.txt 2>&1 && touch logs/iter_done/$id && echo "ok $id $(date +%H:%M)" || { echo "FAIL $id"; exit 1; }; }
for s in "$@"; do case $s in
  it0) step it0_eval ITER=0 NAME=base UNET=wx HEAD=unet,hybrid $PY scripts/26_iter_eval.py ;;
  it1) step it1_pred RUNS=wx VAL_ERA5=1 $PY scripts/25_iter_predict.py
       step it1_eval ITER=1 NAME=tta UNET=wx_tta HEAD=unet,hybrid $PY scripts/26_iter_eval.py ;;
  it3a) step it3a_eval ITER=3a NAME=gbm1 UNET=wx_tta HEAD=gbm $PY scripts/26_iter_eval.py ;;
  it2) step it2_pred RUNS=wxs1,wxs2 VAL_ERA5=1 $PY scripts/25_iter_predict.py
       step it2_eval ITER=2 NAME=ens3 UNET=wx_tta,wxs1_tta,wxs2_tta HEAD=unet,hybrid $PY scripts/26_iter_eval.py ;;
  it3) step it3_eval ITER=3 NAME=ens3gbm UNET=wx_tta,wxs1_tta,wxs2_tta HEAD=gbm $PY scripts/26_iter_eval.py ;;
  it4) step it4_pred RUNS=wxphys VAL_ERA5=1 $PY scripts/25_iter_predict.py
       step it4_eval ITER=4 NAME=phys UNET=wxphys_tta HEAD=unet,hybrid $PY scripts/26_iter_eval.py ;;
  it5) step it5_eval ITER=5 NAME=final UNET=wx_tta,wxbig_tta HEAD=unet,hybrid $PY scripts/26_iter_eval.py
       step it5_export VARIANT=iter NAME=final UNET=wx_tta,wxbig_tta MODELS=wx,wxbig TTA=1 $PY scripts/21_prod_export.py ;;
esac; done
echo "all done $(date +%H:%M)"
