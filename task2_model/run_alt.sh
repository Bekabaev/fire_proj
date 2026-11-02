#!/bin/sh
# Альтернативные подходы (шаги 10–13): потолок/метрики с допуском, решатель, больше крупных срезов, единая оценка с калибровкой.
cd "$(dirname "$0")"
PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"
[ -f out_v3/ceiling.csv ] || $PY scripts/10_ceiling_metrics.py > logs/10_ceiling.txt 2>&1 || exit 1
[ -f cache_v3/trainx_ok ] || $PY scripts/13_prep_big.py > logs/13_prep_big.txt 2>&1 || exit 1
[ -f out_v3/swx_done ] || RUN=swx EPOCHS=12 BS=48 $PY scripts/11_train_solver.py >> logs/11_train_swx.txt 2>&1 || exit 1
[ -f out_v3/wxbig_done ] || RUN=wxbig EXTRA=trainx $PY scripts/07_train_v3.py >> logs/07_train_v3_wxbig.txt 2>&1 || exit 1
$PY scripts/12_eval_all.py > logs/12_eval_all.txt 2>&1 || exit 1
echo done
