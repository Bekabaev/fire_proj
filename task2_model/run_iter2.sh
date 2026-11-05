#!/bin/bash
# Итерации 6-8 (ITERATIONS.md, 05.11 утро). Возобновляемо: шаг отмечается logs/iter_done/<шаг>; обучение само продолжается с *_last.pt.
# Запуск: bash run_iter2.sh; прогресс - logs/run_iter2.txt
cd "$(dirname "$0")"; PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"; mkdir -p logs/iter_done
step() { local id=$1; shift; [ -f logs/iter_done/$id ] && { echo "skip $id"; return 0; }
  echo "== $id $(date +%H:%M)"; env "$@" > logs/iter_$id.txt 2>&1 && touch logs/iter_done/$id && echo "ok $id $(date +%H:%M)" || { echo "FAIL $id"; exit 1; }; }
step it6_train RUN=wxn AUG_GFS=1 VAL_SRC=gfsa $PY scripts/07_train_v3.py
step it6_pred  RUNS=wxn $PY scripts/25_iter_predict.py
step it7_train RUN=wxbign EXTRA=trainx AUG_GFS=1 VAL_SRC=gfsa $PY scripts/07_train_v3.py
step it7_pred  RUNS=wxbign $PY scripts/25_iter_predict.py
step it8_train RUN=wxt AUG_GFS=1 EDGE=0.3 VAL_SRC=gfsa $PY scripts/07_train_v3.py
step it8_pred  RUNS=wxt $PY scripts/25_iter_predict.py
echo "all done $(date +%H:%M)"
