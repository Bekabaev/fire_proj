#!/bin/sh
# Прод (04.11): пересчёт физики с геометрией ячейки по широте -> гибрид (20) -> экспорт (21) -> симулятор (22, 24) -> проверки.
# Готовые шаги не пропускаются (физика пересчитывается вся). Лог каждого шага - logs/<шаг>.txt, итог - logs/run_prod.txt.
cd "$(dirname "$0")"
PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"
REDO=val_era5,val_gfsa,test_era5,test_gfsa,reserves_era5 $PY scripts/19_prod_prep.py > logs/19_prod_prep_geom.txt 2>&1 || exit 1
echo physics ok >> logs/run_prod.txt
$PY scripts/20_prod_hybrid.py > logs/20_prod_hybrid.txt 2>&1 || exit 1
VARIANT=val $PY scripts/21_prod_export.py > logs/21_prod_export.txt 2>&1 || exit 1
$PY scripts/22_prod_sim.py > logs/22_prod_sim.txt 2>&1 || exit 1
$PY scripts/24_prod_sim_speed.py > logs/24_prod_sim_speed.txt 2>&1 || exit 1
echo models ok >> logs/run_prod.txt
(cd .. && ../.venv/Scripts/python.exe -X utf8 -W ignore -m prod.checks > prod/checks_log.txt 2>&1) || exit 1
echo done >> logs/run_prod.txt
