# AGENT_LOG — журнал для агентов Claude Code (win-pc ↔ laptop)

Общий журнал двух агентов Claude Code: на Windows-ПК (`win-pc`) и на ноутбуке с Debian (`laptop`).
Новые записи пишутся сверху. Обновляется через `/sync` (`.claude/skills/sync/SKILL.md`).
Секреты, токены и email сюда не писать.

Формат записи:

```
## YYYY-MM-DD HH:MM · win-pc|laptop
**Сделано:** ...
**Изменённые файлы:** ...
**Коммит:** <hash> <сообщение>
**Для другого агента:** что проверить или сделать дальше, что не трогать
```

---

## 2026-11-08 22:20 · win-pc
**Сделано:** rev1 по правкам пользователя. Презентация `presentation/presentation_final2_rev1.pptx` = его `presentation_final2(2)(1).pptx` + только удаление несостыковок (заметки со ссылками на удалённые слайды, «для обучения» у тестового примера, «карта топлива в разработке / точность близкая» у Ротермела, остаток подписи «Модели · 20») и новые скриншоты сайта с контуром «старт + прогноз» на слайде 25. Отчёт `FireSpread_ML_Report_rev1.docx` = его `_REV` + новый рисунок 5 и рисунок 6 (пожар 1.10.2024 на 4 горизонтах). Скриншоты пересняты headless Edge.
**Изменённые файлы:** presentation/presentation_final2_rev1.pptx, FireSpread_ML_Report_rev1.docx, presentation/site_shots/{site_case4,crop39_*,rep_site_hz}.png

## 2026-11-08 22:10 · win-pc
**Сделано:** сайт — пунктирный контур «общая гарь по прогнозу = старт + прогноз» (поле `total` в horizons, demo + live, легенда); сервер перезапущен, туннель тот же. Скачана правка презентации пользователя `presentation/presentation_final2(2)(1).pptx` (26 слайдов, редактировалась в Google Slides).
**Изменённые файлы:** prod/demo.py, prod/live.py, prod/web/index.html

## 2026-11-08 · win-pc
**Сделано:** по просьбе пользователя из презентаций и отчёта убраны: сайт и Telegram-бот как «продукт» (скриншоты остались как
«Прогноз против факта»), fires.kz, ELMFIRE без настройки — везде только ELMFIRE со скоростью ×0,05 (F1 72 ч 0,19; пример пожара:
62 км², F1 0,40). В заключении отчёта п. 2 и 5 переформулированы (нейросеть — «где и сколько», следующий шаг — отдельная модель
«будет ли сильный рост»).
**Изменённые файлы:** `presentation/presentation_final2.pptx` (31), `presentation_short.pptx` (16), `FireSpread_ML_Report.docx` (15 стр.),
`presentation/tools/make_final2.py`, `make_final2_figs.py`, `make_report.py`, `presentation/final2_figs/*.png`, `AGENT_LOG.md`
**Коммит:** «Decks + report: no site/bot, ELMFIRE x0.05 only» + agent-log
**Для другого агента:** не возвращать в материалы сайт/бота, fires.kz и ELMFIRE без поправки.

## 2026-11-08 · win-pc
**Сделано:** `presentation_final2.pptx` почищен по замечаниям пользователя (теперь 31 слайд): убраны «ИИ против физики» (график),
«Направление и масштаб», чёрный слайд ELMFIRE, «ELMFIRE: точность и площадь», «по часам», оба слайда fires.kz. Вместо них —
4 минималистичных слайда по скиллам pptx + dataviz: «Гибрид против Ротермела» (нативная диаграмма, ×3,6), «Гибрид против ELMFIRE» (×2,9),
«Один пожар за 72 часа» (факт / ELMFIRE / гибрид, `final2_figs/fire_case4.png`), «Что доказано, а что нет» (3 карточки).
`presentation_short.pptx` пересобран с теми же слайдами (16).
**Изменённые файлы:** `presentation/presentation_final2.pptx`, `presentation_short.pptx`, `presentation/tools/make_final2.py`, `make_final2_figs.py`,
`presentation/final2_figs/fire_case4.png`, `AGENT_LOG.md`
**Коммит:** «final2: minimal comparison slides» + agent-log
**Для другого агента:** подпись внизу слайда не начинать с числа — автонумерация его заменит.

## 2026-11-08 · win-pc
**Сделано:** короткая презентация `presentation/presentation_short.pptx` (15 слайдов, только суть; `make_final2.py --short`) и
отчёт `FireSpread_ML_Report.docx` (17 стр.) в формате `Jafar_ML_Report.docx` (его стили, поля, таблицы и подписи), простым языком:
данные, метки, 28 карт + 49 чисел, модели, ИИ против физики (что доказано и что нет), ELMFIRE, fires.kz, ограничения.
Сборка: `presentation/tools/make_report.py`, затем `update_toc.ps1` (Word обновляет содержание).
**Изменённые файлы:** `presentation/presentation_short.pptx`, `FireSpread_ML_Report.docx`, `presentation/tools/make_report.py`,
`update_toc.ps1`, `make_final2.py` (флаг --short), `presentation/final2_figs/report/*.png`, `AGENT_LOG.md`
**Коммит:** «Short deck + report in Jafar format» + agent-log
**Для другого агента:** автор в отчёте — Beibars Kurmanbaev (поменять, если нужно, в make_report.py).

## 2026-11-08 · win-pc
**Сделано:** честная проверка тезиса «ИИ точнее физики» в `presentation_final2.pptx` (теперь 35 слайдов). Доказано: тест 2023–24
(U-Net 0,31 [0,25–0,34] против FBP 0,12 [0,10–0,14], бутстрэп по пожарам, `prod_eval.csv`); ELMFIRE на 40 пожарах — на 48–72 ч
(парное сравнение по пожарам). Не доказано: заповедники 48–72 ч (35 пожаров, интервалы перекрываются), ELMFIRE на 12 ч (8 побед / 10 поражений).
Убран заголовок «×8 к ELMFIRE» (оставлен ×2,9 к ELMFIRE ×0,05), добавлены ×1,7 на экспертах и слайд 30 «что доказано, а что нет»
(фора ИИ от спутниковых меток, упрощённая физика, занижение крупных летних пожаров). Поправлены «Итог», титул, слайд 26 (площадь на экспертах ×1,9–2,0).
**Изменённые файлы:** `presentation/presentation_final2.pptx`, `presentation/tools/make_final2.py`, `AGENT_LOG.md`
**Коммит:** «final2: honest evidence for AI vs physics» + agent-log
**Для другого агента:** не возвращать «×8» как главный тезис — это против ненастроенного ELMFIRE.

## 2026-11-08 · win-pc
**Сделано:** `presentation/presentation_final2.pptx` (34 слайда) — версия для комиссии «ИИ точнее физических моделей, гибрид = ИИ + физика».
Основа — `presentation_final.pptx` (не изменён). Новые слайды: главный вывод (×2,7 к FBP, ×3,7 к Ротермелу на тесте, ×8 к ELMFIRE на 40 пожарах),
метки VIIRS/MODIS/MCD64A1 (доли и веса), 28 карт U-Net, 49 чисел (FiLM), три физические модели, график «ИИ против физики» (тест + заповедники),
ELMFIRE: F1 и площадь, ELMFIRE по часам. Изменены: титул, «Итог», нумерация подписей. Заметки докладчика у всех слайдов.
Источники чисел: ITERATIONS.md (итог итераций 6–8, итерация 9 — Ротермел), CHECKPOINT_task2_model.md §10.8 (FBP), elmfire_bench/results.*.
**Изменённые файлы:** `presentation/presentation_final2.pptx`, `presentation/tools/make_final2.py`, `make_final2_figs.py`, `render_pptx.ps1`,
`presentation/final2_figs/*.png`, `AGENT_LOG.md`
**Коммит:** «Presentation final 2 for the commission» + agent-log
**Для другого агента:** правки делать через `make_final2.py` (пересобирает из presentation_final.pptx); перед правкой pptx — pull.

## 2026-11-08 · win-pc
**Сделано:** по просьбе laptop добавил ELMFIRE ×0,05 в интерактивную визуализацию (https://claude.ai/artifact/9rbDe5ByZmvjgt93scYPzb, версия 2):
пунктирный фиолетовый контур на карте, строка в таблице пожара, столбец в графике F1, третья точка на графике площадей.
**Изменённые файлы:** `elmfire_bench/export_viz.py` (6-й слой — `elmfire_adj0.05_toa.tif`), `AGENT_LOG.md`
**Коммит:** «agent-log: viz with ELMFIRE x0.05»
**Для другого агента:** новые варианты ELMFIRE — пиши в журнал, пересоберу страницу. Дальше по плану HANDOFF §5 (модель «вырастет ли пожар»), ждёт решения пользователя.

## 2026-11-08 16:47 · laptop
**Сделано:** второй прогон ELMFIRE на 40 пожарах с множителем скорости фронта 0,05 (= `k_ros` нашей FBP, подобран на val): F1
12/24/48/72 ч 0,100/0,153/0,198/0,189, площадь ×0,59/0,81/0,91/0,80. Откалиброванный ELMFIRE на 12–24 ч лучше нашей FBP
(0,063/0,101), на 48 ч наравне (0,216), на 72 ч хуже (0,323); гибрид (0,160/0,224/0,360/0,542) лучше обоих. Строка «ELMFIRE ×0,05» и вывод добавлены на слайд 25 (`add_slide.py` теперь заменяет свой слайд при повторном запуске).
**Изменённые файлы:** `elmfire_bench/run_elmfire.py` (`--adj=`), `evaluate.py` (все `elmfire*_toa.tif`), `results.*`,
`cases/*/elmfire_adj0.05_toa.tif`, `PROGRESS.md`
**Коммит:** «ELMFIRE run with spread multiplier 0.05» + этот agent-log
**Для другого агента:** появились новые `cases/*/elmfire_adj0.05_toa.tif` (ELMFIRE со скоростью ×0,05; `elmfire_toa.tif` не менялись) — добавь этот вариант в интерактивную визуализацию. Перед правкой `presentation_final.pptx` сделай pull — слайд ELMFIRE обновлён.

## 2026-11-08 · win-pc
**Сделано:** /sync — подтянул результаты ELMFIRE с ноутбука. Сделал интерактивную визуализацию ELMFIRE / гибрид / FBP / факт
на 40 тестовых пожарах (карта с прокруткой 0–72 ч, ветер, F1 по кейсу, сводные F1 и площадь прогноз/факт):
https://claude.ai/artifact/9rbDe5ByZmvjgt93scYPzb (приватная). Данные собирает `elmfire_bench/export_viz.py <шаблон> <выход>`.
**Изменённые файлы:** `elmfire_bench/export_viz.py`, `AGENT_LOG.md`
**Коммит:** «agent-log: ELMFIRE interactive viz»
**Для другого агента:** шаблон страницы не в git (живёт в артефакте); если меняешь `cases/*/elmfire_toa.tif` — скажи, пересоберу страницу.

## 2026-11-07 19:04 · laptop
**Сделано:** HANDOFF §4 — ELMFIRE 1.1 прогнан на всех 40 тестовых пожарах (сетка 500 м, погода GFS кусочно-постоянная,
старт = зажигания во всех клетках границы стартовой гари). F1 по всем вместе 12/24/48/72 ч: ELMFIRE 0,041/0,047/0,062/0,064;
FBP 0,063/0,101/0,216/0,323; гибрид 0,160/0,224/0,360/0,542. Площадь ELMFIRE ×19–24 (упор в окно 48 км): 30 из 40 пожаров
> 5× факта, 4 без роста (влажное/зелёное топливо). Направление по ветру верное. Гипотеза HANDOFF подтверждена.
Слайд добавлен в `presentation_final.pptx` (25-й из 26, перед «Итогом», подпись «Результаты · 27»).
**Изменённые файлы:** `elmfire_bench/` (`run_elmfire.py`, `make_figs.py`, `add_slide.py`, `setup_conda.sh`, `results.*`,
`fig_elmfire_cases.png`, `cases/*/elmfire_toa.tif`), `PROGRESS.md` (строка хронологии), `presentation/presentation_final.pptx`
**Коммит:** «ELMFIRE benchmark: 40 test fires run, results, figure, slide 25» + этот agent-log
**Для другого агента:** `setup_debian.sh` на Debian не проходит без sudo — рабочий путь `setup_conda.sh`. Начальный периметр
через phi в ELMFIRE 1.1 не даёт фронта (огонь не идёт) — не возвращать. Если правишь `presentation_final.pptx` на Windows,
сначала сделай pull: слайд ELMFIRE уже в нём (бинарный файл, конфликт не смержится). Дальше по HANDOFF §5 (модель «вырастет ли пожар»).

## 2026-11-07 · win-pc
**Сделано:** Заведён канал связи между агентами: этот журнал, скилл `/sync`, `CLAUDE.md`.
**Изменённые файлы:** `AGENT_LOG.md`, `.claude/skills/sync/SKILL.md`, `CLAUDE.md`
**Коммит:** см. `git log` — «Agent sync channel»
**Для другого агента:** При старте сессии запусти `/sync`. Контекст проекта — в `HANDOFF.md`; продолжай с его раздела 4.
