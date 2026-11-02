#!/bin/sh
# Пересборка датасета задания 1 (запуск из папки project_till_nov/task1_dataset).
# Погода кэшируется в data/weather_hourly, кубы пропускаются, если уже есть (03_cubes.py --rebuild = перезаписать).
set -e
PY="../../.venv/Scripts/python.exe -X utf8"
mkdir -p logs
$PY scripts/01_events.py      > logs/01_events.txt
$PY scripts/02_weather.py     > logs/02_weather.txt
$PY scripts/03_cubes.py       > logs/03_cubes.txt
$PY scripts/04_pixel_table.py > logs/04_pixel_table.txt
$PY scripts/05_qa.py          > logs/05_qa.txt
# динамика (нужен ключ FIRMS в файле .firms_key; термоточки кэшируются в data/firms)
$PY scripts/06_firms.py       > logs/06_firms.txt
$PY scripts/07_arrival.py     > logs/07_arrival.txt
$PY scripts/08_slices.py      > logs/08_slices.txt
$PY scripts/09_qa_dynamics.py > logs/09_qa_dynamics.txt
echo done
