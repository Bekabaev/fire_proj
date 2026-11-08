# -*- coding: utf-8 -*-
"""Общая презентация проекта: датасет (слайды шага 31) + снимки Landsat / Sentinel-2 + модели и формулы + результаты +
сравнение с индустрией + симуляции -> project_till_nov/FireSpread_Full_Presentation.pptx. Цифры берутся из CSV task2_model/out_v3,
поэтому после переобучения на метках v4 презентацию достаточно пересобрать."""
import os, shutil, json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
T2 = HERE.parents[1] / "task2_model"; O3 = T2 / "out_v3"; KZD = HERE.parent / "data" / "ext_kz"
src = (HERE / "31_presentation_pptx.py").read_text(encoding="utf8")
src = src.replace('OUT = ROOT.parent / "FireSpread_Dataset_Presentation.pptx"', 'OUT = ROOT.parent / "FireSpread_Full_Presentation.pptx"')
src = src.replace('"ПРОЕКТ FIRESPREAD · ЗАДАНИЕ 1"', '"ПРОЕКТ FIRESPREAD"')
src = src.replace('"Датасет для ML-моделирования распространения степных пожаров"', '"Прогноз распространения степных пожаров: данные, модели, симуляции"')
src = src.replace('"Октябрь 2026"', '"Ноябрь 2026"')
src = src.replace('notes(s, "Презентация только о датасете (задание 1). Модели (задание 2) здесь не рассматриваются.")',
                  'notes(s, "Три части: датасет (с уточнением по снимкам Landsat / Sentinel-2), модели и их качество, симуляции.")')
src = src.replace("prs.save(OUT)\nprint(OUT)", "")
__file__ = str(HERE / "31_presentation_pptx.py")
exec(compile(src, "31_presentation_pptx.py", "exec"))

# рисунки задания 2 -> figures/pres2
P2 = FIG / "pres2"; P2.mkdir(exist_ok=True)
for f in ["pres/f1_models.png", "pres/wind_sensitivity.png", "pres/imagery_labels.png", "sim/phys_demo.png", "sim/burn_probability.png"]:
    if (T2 / "figures" / f).exists():
        shutil.copy(T2 / "figures" / f, P2 / Path(f).name)
reals = sorted((T2 / "figures" / "sim").glob("real_*.png"))
if reals:
    shutil.copy(reals[-1], P2 / "real_fire.png")
have = lambda n: (P2 / n).exists()


def section(num, head, sub):
    s = prs.slides.add_slide(BLANK); bg(s, DARK)
    text(s, 0.8, 2.3, 11.5, 0.5, num, size=18, color=OCHRE, bold=True)
    text(s, 0.8, 2.85, 11.5, 1.2, head, size=40, font=HEAD, bold=True, color=WHITE)
    text(s, 0.8, 4.2, 11.0, 1.0, sub, size=18, color=RGBColor(0xE8, 0xDF, 0xD4))
    return s


def f2(x):
    return f"{x:.2f}".replace(".", ",")


ev = pd.read_csv(O3 / "eval_all.csv")
evi = pd.read_csv(O3 / "eval_all_img.csv") if (O3 / "eval_all_img.csv").exists() else None
pb = pd.read_csv(O3 / "phys_baseline.csv") if (O3 / "phys_baseline.csv").exists() else None
cmp_ = pd.read_csv(O3 / "comparison_v3.csv"); cmp_ = cmp_[cmp_.subset == "all"]


def f1row(df, model, part):
    r = df[(df.model == model) & (df.part == part)].sort_values("horizon_h")
    return [f2(v) for v in r.F1.values] if len(r) == 4 else ["—"] * 4


# ================================================================ II. новые данные: снимки
section("ЧАСТЬ II", "Уточнение меток по снимкам Landsat 8/9 и Sentinel-2",
        "Контуры гари 10–30 м между пролётами MODIS / VIIRS")
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Как снимки уточняют время прихода огня", "Planetary Computer: Landsat 8/9 C2 L2 (30 м, 8 сут) и Sentinel-2 L2A (20 м, 5 сут)")
qa = (KZD / "qa_labels_img.txt").read_text(encoding="utf8").splitlines() if (KZD / "qa_labels_img.txt").exists() else []
thr = qa[0] if qa else ""
bullets(s, 0.6, 1.6, 12.1, 2.6, [
    "Готовый продукт USGS Landsat Burned Area есть только для США — гарь распознаётся по dNBR сами",
    "NBR = (NIR − SWIR2) / (NIR + SWIR2); dNBR = NBR до пожара − NBR снимка, поправка на сезонное высыхание по фону",
    "Порог подобран по ячейкам с известным состоянием (VIIRS): " + thr.replace("dNBR threshold", "").strip(),
    "Метка сдвигается раньше, если гарь уже видна на снимке; добавляются ячейки гари, пропущенные MCD64A1",
    "Позже не сдвигаем: в степи сигнал гари слабый и пропадает при отрастании травы",
    "5 024 пожара ≥ 10 км² (сначала тест и val); метки v4 → переобучение моделей",
], size=13, space=3)
if have("imagery_labels.png"):
    pic(s, "pres2/imagery_labels.png", 0.6, 4.25, 12.1, 3.15)
notes(s, "Снимки дают контур, но не время: время зажимается между снимками. Шаг 33 - скачивание и dNBR, шаг 34 - метки, шаг 16 - кэш.")

# ================================================================ III. модели
section("ЧАСТЬ III", "Модели прогноза", "Ориентиры, U-Net с погодой, «сеть → скорость → решатель», ансамбль")
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Задача и ориентиры", "Вход: состояние пожара и среда на момент t. Выход: вероятность «впервые сгорит» к 12 / 24 / 48 / 72 ч")
for k, (head, col, items) in enumerate([
        ("Расширение гари", GREY, ["Текущая гарь, расширенная на r ячеек", "r подбирается по val", "Нижняя планка: модель должна быть заметно лучше"]),
        ("Физика FBP O-1b", OCHRE, ["Канадская система для открытой степи", "ROS = 250·(1 − e^(−0,035·ISI))^1,7 · CF", "Эллипс по ветру, склон, топливо из WorldCover", "Один параметр k (множитель скорости) по val"]),
        ("Нейросети", EMBER, ["U-Net с погодой (FiLM)", "Сеть → скорость → решатель", "Ансамбль двух моделей", "Калибровка вероятностей по val"])]):
    x = 0.6 + k * 4.1
    card(s, x, 1.85, 3.8, 4.9)
    text(s, x + 0.3, 2.05, 3.3, 0.6, head, size=19, font=HEAD, bold=True, color=col)
    bullets(s, x + 0.3, 2.8, 3.3, 3.8, items, size=14, space=8)

s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "U-Net с погодой (FrontV3)", "~541 тыс. параметров; 28 карт 96 × 96 + 49 чисел (сезон, состояние, погода)")
text(s, 0.6, 1.7, 6.0, 0.4, "Устройство", size=18, font=HEAD, bold=True, color=EMBER)
bullets(s, 0.6, 2.2, 6.0, 2.5, [
    "4 уровня 16 → 32 → 64 → 128 каналов: левая половина сжимает и видит контекст, правая возвращает детали",
    "Погода входит через FiLM: признак × (1 + γ) + β, γ и β — из 49 чисел",
    "4 вложенных выхода: вероятность к 72 ч не меньше, чем к 24 ч"], size=14, space=8)
card(s, 6.9, 1.7, 5.8, 2.9)
text(s, 7.2, 1.85, 5.3, 2.7, [
    [("Накопленные логиты", {"bold": True, "color": EMBER})],
    "L12 = o1",
    "L24 = L12 + softplus(o2)",
    "L48 = L24 + softplus(o3)",
    "L72 = L48 + softplus(o4)",
    "p = sigmoid(L),  softplus(x) = ln(1 + eˣ) > 0"], size=15, font="Consolas")
text(s, 0.6, 4.85, 12.1, 0.4, "Обучение", size=18, font=HEAD, bold=True, color=EMBER)
bullets(s, 0.6, 5.3, 12.1, 2.0, [
    "Лосс BCE (pos_weight 5) + Dice; вес ячейки по точности метки 1 / (1 + неопределённость / 12 ч); на 12 ч «только дата» не учитывается",
    "Аугментация: повороты / отражения (ветер поворачивается вместе с картой), вырезка 80 × 80; AdamW + OneCycle, 24 эпохи",
    "Порог по F1 и изотоническая калибровка вероятностей — на val 2022"], size=14, space=6)

s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Сеть → скорость → решатель (SolverNet)", "Сеть предсказывает скорость огня, фронт продвигает дифференцируемый «навигатор»")
card(s, 0.6, 1.7, 6.0, 4.9)
text(s, 0.9, 1.85, 5.5, 4.7, [
    [("1. Скорость в 16 направлениях", {"bold": True, "color": SAGE})],
    "ln S_k = A + 0,5·tanh(C_k) + B·(e_k · w)",
    "",
    [("2. Время перехода", {"bold": True, "color": SAGE})],
    "τ_k = 0,5 км · |шаг| / S_k",
    "",
    [("3. Решатель, 48 итераций", {"bold": True, "color": SAGE})],
    "T(x) = min( T(x), min_k [T(x − k) + τ_k] )",
    "",
    [("4. Вероятность", {"bold": True, "color": SAGE})],
    "p(h) = sigmoid((h − T) / σ),  σ ≈ 14 ч"], size=14, font="Consolas")
bullets(s, 6.9, 1.8, 5.8, 4.9, [
    "A — базовая скорость ячейки, C_k — поправка направления, B — влияние ветра w (окно 24 ч)",
    "Старт с кромки: задержка softplus(o) — кромка может быть потухшей",
    "Физический смысл и нет ограничения радиусом; площадь на экспертных пожарах 0,9–1,0 от факта",
    "σ ≈ 14 ч — модель сама оценила неопределённость времени прихода",
    "Обучение в ~8 раз дольше U-Net; на тесте GFA слабее, в ансамбле — лучший результат"], size=14, space=9)

# ================================================================ результаты
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Качество прогноза", "Строгий F1 по новой гари; тест — честный режим с прогнозом погоды GFS, выпущенным за 12 ч до t")
if have("f1_models.png"):
    pic(s, "pres2/f1_models.png", 0.4, 1.55, 12.5, 4.4)
rows = [["", "12 ч", "24 ч", "48 ч", "72 ч"],
        ["Расширение гари (тест)"] + [f2(v) for v in cmp_[(cmp_.variant == "ring") & (cmp_.part == "test")].sort_values("horizon_h").F1.values]]
if pb is not None and len(pb[(pb.part == "test") & (pb.src == "gfsa")]) == 4:
    rows.append(["Физика FBP (тест, GFS)"] + [f2(v) for v in pb[(pb.part == "test") & (pb.src == "gfsa")].sort_values("horizon_h").F1.values])
rows += [["U-Net с погодой (тест, GFS)"] + f1row(ev, "wx_gfsa", "test"), ["Ансамбль (эксперты)"] + f1row(ev, "ens[wx_era5+swx_era5]", "reserves")]
table(s, 0.6, 6.0, 12.1, rows, [4.5, 1.9, 1.9, 1.9, 1.9], size=12, row_h=0.28)

s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Сравнение с индустрией", "Прямого эталона для степей Казахстана нет — ближайшие опубликованные постановки")
table(s, 0.6, 1.75, 12.1, [
    ["Работа", "Задача", "Результат"],
    ["Next Day Wildfire Spread (Huot et al., 2022)", "активный огонь завтра, 1 км, США", "F1 ≈ 0,38; PR-AUC 0,28 (персистентность 0,12)"],
    ["WildfireSpreadTS (2023)", "активный огонь завтра, 375 м", "AP ≈ 0,34–0,38 (персистентность ≈ 0,19)"],
    ["Cruz & Alexander (2013)", "физические модели скорости", "ошибка скорости 51–75 %; ±35 % — хороший уровень"],
    ["Наша модель (тест 2023–24, GFS)", "новая гарь 12–72 ч, 500 м", "F1 0,23–0,29; с допуском 1 ячейка 0,34–0,43"],
    ["Наш ансамбль (эксперты)", "новая гарь 12–72 ч, 500 м", "F1 0,29–0,47 при потолке разметки ≈ 0,58"],
], [4.3, 3.6, 4.2], size=13, row_h=0.55, hl_rows=(4, 5))
bullets(s, 0.6, 5.3, 12.1, 1.8, [
    "В NDWS-постановках в ответ входят уже горящие ячейки (их легко угадать) — у нас только новая гарь, это строже",
    "Две независимые разметки одного пожара совпадают лишь на F1 ≈ 0,58: модель не может быть точнее «правды»"], size=14, space=8)

s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, "Почему не 60–90 %", "Предел задают данные и сама задача, а не архитектура")
for k, (big, lab, col) in enumerate([
        ("≈ 0,58", "потолок: совпадение экспертной\nразметки и GFA на тех же пожарах", EMBER),
        ("0,29 → 0,43", "F1 на 72 ч при допуске 1 ячейка:\nновая гарь — тонкая полоса", SAGE),
        ("σ ≈ 14 ч", "неопределённость времени прихода:\nсмены ветра, тушение, барьеры", DARK),
        ("+0,01", "столько даёт идеальная погода (ERA5)\nвместо прогноза GFS", OCHRE)]):
    x = 0.6 + (k % 2) * 6.15; y = 1.75 + (k // 2) * 2.6
    card(s, x, y, 5.95, 2.35)
    text(s, x + 0.3, y + 0.25, 5.4, 0.9, big, size=34, font=HEAD, bold=True, color=col, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, y + 1.2, 5.4, 1.0, lab.split("\n"), size=15)

if evi is not None:
    s = prs.slides.add_slide(BLANK); bg(s, WHITE)
    title(s, "Проверка на метках, уточнённых по снимкам", "Те же модели; калибровка и порог — на val с метками v4. Переобучение на метках v4 — run_v4.sh")
    rows = [["Модель / часть", "12 ч", "24 ч", "48 ч", "72 ч"]]
    for lab, m, part, d in (("U-Net, тест GFS — старые метки", "wx_gfsa", "test", ev), ("U-Net, тест GFS — метки v4", "wx_gfsa", "test", evi),
                            ("Решатель, тест GFS — старые метки", "swxh_gfsa", "test", ev), ("Решатель, тест GFS — метки v4", "swxh_gfsa", "test", evi),
                            ("Ансамбль, тест GFS — метки v4", "ens[wx_gfsa+swxh_gfsa]", "test", evi)):
        rows.append([lab] + f1row(d, m, part))
    table(s, 0.6, 1.8, 12.1, rows, [5.3, 1.7, 1.7, 1.7, 1.7], size=14, row_h=0.5, hl_rows=(2, 4))

# ================================================================ IV. симуляции
section("ЧАСТЬ IV", "Симуляции", "Физический эталон, чувствительность моделей к ветру, повтор реальных пожаров, вероятность выгорания")
if have("phys_demo.png"):
    s = prs.slides.add_slide(BLANK); bg(s, WHITE)
    title(s, "Физический симулятор на синтетических сценах", "FBP O-1b + эллипс по ветру + кратчайшее время прихода; k подобран так, чтобы площадь совпадала с фактом")
    pic(s, "pres2/phys_demo.png", 0.4, 1.5, 7.9, 5.8)
    bullets(s, 8.5, 1.8, 4.4, 5.2, [
        "Ветер вытягивает пожар в эллипс: L/B по формуле FBP (2,3 при 18 км/ч)",
        "Река и зелёная пашня останавливают фронт",
        "Вверх по склону — быстрее (множитель FBP)",
        "Тот же симулятор — в интерактивной странице fire_simulator.html: ветер, сухость, ландшафт, поджог кликом"], size=14, space=10)
if have("wind_sensitivity.png"):
    s = prs.slides.add_slide(BLANK); bg(s, WHITE)
    title(s, "Тест чувствительности: подменяем только ветер", "60 реальных срезов теста; ветер 8 м/с с четырёх сторон и штиль; всё остальное без изменений")
    pic(s, "pres2/wind_sensitivity.png", 0.5, 1.7, 6.6, 4.4)
    ws = pd.read_csv(O3 / "sim_wind_sensitivity.csv")
    a = ws.groupby(["model", ws.wind == "calm"]).area_km2.median().unstack()
    bullets(s, 7.4, 1.8, 5.4, 5.2, [
        "Физика смещает прогноз по ветру в 86 % случаев, нейросети — в 29–36 % (случайно ≈ 25 %)",
        f"Сила ветра нейросетями учтена: площадь 72 ч при 8 м/с больше, чем в штиль (U-Net {a.loc['wx', True]:.0f} → {a.loc['wx', False]:.0f} км², "
        f"решатель {a.loc['swxh', True]:.0f} → {a.loc['swxh', False]:.0f} км²)",
        "Направление модели берут из формы и активной кромки пожара: ветер ERA5 по ячейке 0,25° — слабый сигнал",
        "Это не ошибка аугментации (проверено) — следующий шаг: физические ограничения / ветер в решателе на каждом шаге"], size=14, space=9)
if have("real_fire.png"):
    s = prs.slides.add_slide(BLANK); bg(s, WHITE)
    title(s, "Повтор реального пожара из теста", "Факт, U-Net, решатель и физика FBP; контуры 12 / 24 / 48 / 72 ч")
    pic(s, "pres2/real_fire.png", 0.3, 1.7, 12.7, 3.6)
    bullets(s, 0.6, 5.5, 12.1, 1.7, [
        "Физика ведёт огонь по ветру, а пожар ушёл поперёк — сменился ветер у кромки, рельеф, тушение",
        "Нейросети осторожны: прогнозируют рост там, где кромка активна, площадь занижают"], size=14, space=8)
if have("burn_probability.png"):
    s = prs.slides.add_slide(BLANK); bg(s, WHITE)
    title(s, "Карта вероятности выгорания", "100 случайных возгораний на реальном ландшафте с реальной погодой (как Burn-P3 / FSim)")
    pic(s, "pres2/burn_probability.png", 0.6, 1.6, 6.0, 5.4)
    bullets(s, 7.0, 1.9, 5.8, 5.0, [
        "Доля прогонов, в которых ячейка сгорела за 72 ч",
        "Видно, где сплошная сухая трава проводит огонь, а где пашня и вода его останавливают",
        "Такие карты используют для планирования противопожарных разрывов и патрулей",
        "Нейросети на искусственных поджогах не работают: такие сцены вне распределения обучения — для этого нужен физический симулятор"], size=14, space=10)

# ================================================================ итоги
s = prs.slides.add_slide(BLANK); bg(s, DARK)
text(s, 0.6, 0.45, 12.1, 0.8, "Итоги и следующие шаги", size=32, font=HEAD, bold=True, color=WHITE)
for k, (head, items) in enumerate([
        ("Сделано", ["Датасет 111 тыс. срезов + экспертные пожары", "Метки уточнены по Landsat / Sentinel-2",
                     "U-Net, решатель, ансамбль: в 2–5 раз лучше расширения гари и физики", "Симулятор FBP, тест чувствительности, интерактивная страница"]),
        ("Дальше", ["Переобучение на метках v4 (run_v4.sh)", "Ветер в решателе на каждом шаге — учёт направления",
                    "Предобучение на синтетических пожарах (sim-to-real)", "Соседние страны (Россия, Монголия): +50–70 % крупных пожаров"])]):
    x = 0.6 + k * 6.25
    text(s, x, 1.65, 5.8, 0.5, head, size=20, font=HEAD, bold=True, color=OCHRE)
    bullets(s, x, 2.3, 5.8, 3.6, items, size=16, color=RGBColor(0xF2, 0xEC, 0xE4), space=12)
text(s, 0.6, 6.2, 12.1, 0.6, "Подробно: ОБЪЯСНЕНИЕ_на_пальцах.md, CHECKPOINT_task2_model.md", size=14, color=OCHRE, italic=True)

prs.save(OUT)
print(OUT, len(prs.slides), "slides")
