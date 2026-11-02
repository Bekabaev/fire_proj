#!/bin/sh
# Задание 2: обучение и сравнение моделей (запуск из папки project_till_nov/task2_model).
# Перед этим должны быть готовы срезы Казахстана: task1_dataset/scripts 10–15 (см. PROGRESS.md).
set -e
PY="../../.venv/Scripts/python.exe -X utf8"
mkdir -p logs
$PY scripts/01_unet.py     > logs/01_unet.txt       # U-Net без погоды (GPU, ~30 эпох)
$PY scripts/02_physics.py  > logs/02_physics.txt    # физика: время прихода огня по Ротермелу
$PY scripts/03_hybrid.py   > logs/03_hybrid.txt     # GBM-поправки, гибрид, сравнение на тесте 2023–2024
$PY scripts/04_reserves.py > logs/04_reserves.txt   # проверка на экспертных контурах заповедников
$PY scripts/05_figures.py  > logs/05_figures.txt
echo done
