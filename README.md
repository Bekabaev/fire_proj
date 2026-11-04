# Fire spread — прогноз распространения пожаров в Казахстане

Прогноз новой гари на 12 / 24 / 48 / 72 ч по состоянию пожара, растительности, рельефу и погоде (ERA5 / прогноз GFS).

- `task1_dataset/` — сборка датасета (GFA + FIRMS + MCD64A1, WorldCover, GLO-90, MOD13A1, OSM, ERA5, GFWED, GFS). Скрипты: `scripts/`, запуск: `run_v3.sh`.
- `task2_model/` — модели (U-Net FrontV3 с FiLM, SolverNet «сеть → скорость → решатель», ансамбль), оценка. Запуск: `run_v3.sh`, `run_alt.sh`.
- `task2_model/out_v3/` — метрики (`eval_all.csv`, `ceiling.csv`, `tolerant_metrics.csv`) и лучшие чекпойнты моделей (`*_best.pt`).

Документы:
- **`DOCUMENTATION.md` — полная техническая документация: данные, код, модели, прод-движок, все результаты (начинать с неё)**;
- `ОБЪЯСНЕНИЕ_на_пальцах.md` — объяснение данных, моделей, формул и результатов простым языком;
- `PROGRESS.md`, `CHECKPOINT_task1_dataset.md`, `CHECKPOINT_task2_model.md` — журнал работ;
- `FireSpread_Dataset_Presentation.pptx`, `FireSpread_Dataset_Report.docx` — презентация и отчёт по датасету.

## Чего нет в репозитории
Датасет (`task1_dataset/data/`, ~26 ГБ), кэш обучения (`task2_model/cache_v3/`, ~36 ГБ), сохранённые вероятности (`probs_*.npy`) и промежуточные чекпойнты — слишком велики для GitHub; пересобираются скриптами.
Ключ NASA FIRMS (MAP_KEY) кладётся локально в `task1_dataset/.firms_key` и в репозиторий не входит.
