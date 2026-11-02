#!/bin/sh
# Датасет v3 (шаги 18–30), запуск из project_till_nov/task1_dataset. Нужны готовые шаги 10–15 (данные Казахстана).
# Загрузки кэшируются: повторный запуск докачивает только недостающее.
# Open-Meteo (шаги 13, 21) ограничивает ~10 тыс. запросов в сутки: при «Daily API request limit exceeded»
# запустить этот файл снова на следующий день - шаг 13/21 продолжит с места остановки, шаги 23 и 26 пересчитаются.
set -e
PY="../../.venv/Scripts/python.exe -X utf8"
mkdir -p logs
$PY scripts/13_weather_kz.py          > logs/13_weather_kz_v3.txt      # докачка ERA5 (2014–2020)
$PY scripts/18_arrival_kz.py          > logs/18_arrival_kz.txt         # время прихода огня по VIIRS/MODIS
$PY scripts/19_fuel_context_kz.py     > logs/19_fuel_context_kz.txt    # аномалия NDVI, пик прошлого года, давность гари, OSM
$PY scripts/20_stations_kz.py         > logs/20_stations_kz.txt        # станции NOAA ISD
$PY scripts/21_forecast_archive_kz.py > logs/21_forecast_archive_kz.txt # прогнозы GFS / ECMWF с заблаговременностью (2024)
$PY scripts/22_slices_v3_kz.py        > logs/22_slices_v3_kz.txt       # срезы 12/24/48/72 ч (+ сцены 1 км)
$PY scripts/23_slice_weather_v3_kz.py > logs/23_slice_weather_v3_kz.txt
$PY scripts/24_qa_labels_v3.py        > logs/24_qa_labels_v3.txt       # проверка меток по экспертным контурам
$PY scripts/25_pre_fuel_landsat.py    > logs/25_pre_fuel_landsat.txt   # топливо до пожара для кубов заповедников
$PY scripts/30_perims_imagery.py      > logs/30_perims_imagery.txt     # промежуточные контуры по Sentinel-2/Landsat, уточнение времени
$PY scripts/27_slices_v3_reserves.py > logs/27_slices_v3_reserves.txt  # срезы v3 экспертных пожаров заповедников
RESERVES=1 $PY scripts/23_slice_weather_v3_kz.py > logs/23_slice_weather_v3_reserves.txt
$PY scripts/28_fwi_gfwed.py           > logs/28_fwi_gfwed.txt          # индексы FWI (NASA GFWED) -> колонки gfwed_*
$PY scripts/29_gfs_archive_aws.py     > logs/29_gfs_archive_aws.txt    # прогнозы GFS 2022–2024 с AWS -> колонки fc_gfsa_*
$PY scripts/26_qa_v3.py               > logs/26_qa_v3.txt
echo done
