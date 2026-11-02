#!/bin/sh
# Задание 2 на датасете v3: кэш входов -> U-Net без погоды -> U-Net с погодой -> оценка.
# Можно перезапускать после сбоя: готовые части кэша пропускаются, обучение продолжается с последней сохранённой эпохи.
cd "$(dirname "$0")"
PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"
mkdir -p logs
$PY scripts/06_prep_v3.py >> logs/06_prep_v3.txt 2>&1 || exit 1
for R in nowx wx; do
  [ -f out_v3/${R}_done ] || RUN=$R $PY scripts/07_train_v3.py >> logs/07_train_v3_$R.txt 2>&1 || exit 1
done
$PY scripts/08_eval_v3.py > logs/08_eval_v3.txt 2>&1 || exit 1
$PY scripts/09_paired_v3.py > logs/09_paired_v3.txt 2>&1 || exit 1
echo done
