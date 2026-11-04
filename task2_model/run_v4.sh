#!/bin/sh
# Метки v4 (Landsat / Sentinel-2): ждём окончания скачивания снимков (шаг 33), уточняем метки (34), переносим в кэш (16),
# переобучаем U-Net (wximg) и решатель (swxhimg) на новых метках, оцениваем на старых и на новых метках.
cd "$(dirname "$0")"
PY="../../.venv/Scripts/python.exe -X utf8 -W ignore"
until grep -q "^done" ../task1_dataset/logs/33_imagery_kz.log; do sleep 300; done
(cd ../task1_dataset && $PY scripts/34_labels_imagery_kz.py > logs/34_labels_imagery_kz.log 2>&1) || exit 1
$PY scripts/16_patch_labels.py > logs/16_patch_labels.txt 2>&1 || exit 1
[ -f out_v3/wximg_done ] || RUN=wximg LABELS=img $PY scripts/07_train_v3.py >> logs/07_train_wximg.txt 2>&1 || exit 1
[ -f out_v3/swxhimg_done ] || RUN=swxhimg LABELS=img EPOCHS=12 BS=48 $PY scripts/11_train_solver.py >> logs/11_train_swxhimg.txt 2>&1 || exit 1
LABELS=img RUNS=wx,swxh,wximg,swxhimg $PY scripts/12_eval_all.py > logs/12_eval_img.txt 2>&1 || exit 1
OUTSUF=_v4models RUNS=wx,swxh,wximg,swxhimg $PY scripts/12_eval_all.py > logs/12_eval_v4models.txt 2>&1 || exit 1
echo done
