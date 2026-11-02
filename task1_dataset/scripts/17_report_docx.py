"""Шаг 17. Отчёт по датасету (задание 1) в формате .docx в оформлении FireSpread_Report_2 (стили берутся из него как из шаблона).
Выход: project_till_nov/FireSpread_Dataset_Report.docx
"""
import os, sys, json, copy
import numpy as np, pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

PROJ = os.path.join(ROOT, "project_till_nov")
OUTF = os.path.join(PROJ, "FireSpread_Dataset_Report.docx")
KZ = os.path.join(DATA, "ext_kz")
ACCENT = "1F4E79"; ZEBRA = "EEF3F8"

ev = pd.read_csv(os.path.join(DATA, "events_spread.csv"))
fs = pd.read_csv(os.path.join(DATA, "firms_summary.csv")); ar = pd.read_csv(os.path.join(DATA, "arrival_summary.csv"))
si1 = pd.read_csv(os.path.join(DATA, "slices_index.csv")); pix_cols = json.load(open(os.path.join(DATA, "pixel_samples_columns.json")))
ci = pd.read_csv(os.path.join(DATA, "cubes_index.csv")) if os.path.exists(os.path.join(DATA, "cubes_index.csv")) else None
K = json.load(open(os.path.join(KZ, "kz_stats.json"), encoding="utf8"))
V3 = json.load(open(os.path.join(KZ, "slices_v3_stats.json"), encoding="utf8"))
QW = pd.read_csv(os.path.join(KZ, "qa_weather_v3.csv")).set_index(["source", "var"])
CH3 = json.load(open(os.path.join(KZ, "slices_v3_channels.json"), encoding="utf8"))["channels"]
QL = pd.read_csv(os.path.join(KZ, "qa_labels_v3.csv")) if os.path.exists(os.path.join(KZ, "qa_labels_v3.csv")) else None

# ---------- документ на основе шаблона отчёта
doc = Document(os.path.join(PROJ, "FireSpread_Report_2.docx"))
body = doc.element.body
for el in list(body):
    if el.tag != qn("w:sectPr"):
        body.remove(el)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor); tcPr.append(sh)


def borders(table, color="BFBFBF"):
    tblPr = table._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), color); b.append(e)
    tblPr.append(b)


def para(text="", style=None, bold=False, size=None, italic=False, align=None, space_after=None, color=None):
    p = doc.add_paragraph(style=style)
    if text:
        add_runs(p, text, bold=bold, size=size, italic=italic, color=color)
    if align:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def add_runs(p, text, bold=False, size=None, italic=False, color=None):
    """Текст с выделением **жирным**."""
    parts = text.split("**")
    for i, t in enumerate(parts):
        if not t:
            continue
        r = p.add_run(t); r.bold = bold or (i % 2 == 1); r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


def bullets(items):
    for t in items:
        p = doc.add_paragraph(style="List Bullet") if "List Bullet" in [s.name for s in doc.styles] else doc.add_paragraph()
        if p.style.name != "List Bullet":
            t = "• " + t
        add_runs(p, t); p.paragraph_format.space_after = Pt(2)


def _numlike(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    t = str(v).replace(" ", "").replace(" ", "").replace(",", "").replace(".", "").replace("%", "").replace("−", "").replace("/", "")
    return t.isdigit()


def table(rows, header, widths=None, size=9):
    t = doc.add_table(rows=1, cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; borders(t)
    trPr = t.rows[0]._tr.get_or_add_trPr(); h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)   # шапка на каждой странице
    for j, hd in enumerate(header):
        c = t.rows[0].cells[j]; c.text = ""; add_runs(c.paragraphs[0], hd, bold=True, size=size, color="FFFFFF"); shade(c, ACCENT)
    numcol = [j > 0 and all(_numlike(r[j]) for r in rows) for j in range(len(header))]
    for i, r in enumerate(rows):
        row = t.add_row(); cs = OxmlElement("w:cantSplit"); row._tr.get_or_add_trPr().append(cs); cells = row.cells
        for j, v in enumerate(r):
            cells[j].text = ""; p = cells[j].paragraphs[0]; add_runs(p, str(v), size=size)
            if numcol[j]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if i % 2 == 1:
                shade(cells[j], ZEBRA)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, caption, width=17.5):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(path, width=Cm(width))
    c = doc.add_paragraph(style="Caption"); add_runs(c, caption); c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def fmt(x, d=0):
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s


F = lambda n: os.path.join(FIG, n)

# ---------- титул
para("Датасет для прогноза распространения пожаров", style="Title")
para("Задание 1 · данные ДЗЗ и метеорологии для обучения ML-модели    22 октября 2026", style="Subtitle")
para(f"Собран датасет из двух частей. **Экспертная часть** — 44 пожара четырёх заповедников (2014–2023) с контурами, "
     f"нарисованными специалистами по снимкам 10–30 м, временем прихода огня по термоточкам и полным набором факторов: топливо, рельеф, "
     f"погода, индексы пожарной опасности. **Расширенная часть** — {fmt(K['fires'])} пожаров всего Казахстана за 2013–2024 годы из "
     f"Global Fire Atlas, {fmt(K['slices'])} срезов для обучения нейросети. Обе части приведены к формату прошлого эксперимента "
     f"(FireSpread_Report_2): сцена 96 × 96 ячеек по 500 м в EPSG:6933, прогноз нового выгорания через 24, 48 и 72 часа. "
     f"**Версия v3** расширенной части уточняет время прихода огня по VIIRS до часов, добавляет прогноз на 12 ч, метеостанции, "
     f"оперативные прогнозы и сведения о растительности: {fmt(V3['slices'])} срезов по {fmt(V3['fires'])} пожарам.")

doc.add_heading("Задача и состав датасета", level=1)
para("Модель получает состояние уже обнаруженного пожара и должна оценить, какая территория выгорит в следующие 24, 48 и 72 часа. "
     "Для этого каждому пожару нужно знать: где и когда он начался, как фронт двигался во времени, чем покрыта территория вокруг "
     "(тип и состояние растительности, прошлые гари, рельеф) и какая была погода.")
table([["Экспертная (заповедники)", "44", "235", "2014–2023", "контур эксперта 30 м + время прихода огня по FIRMS/MCD64A1",
        "калибровка физики, честный тест"],
       ["Расширенная (Казахстан, GFA)", fmt(K["fires"]), fmt(K["slices"]), K["years"], "дата выгорания MCD64A1 (сутки), 500 м",
        "обучение U-Net и поправок"],
       ["Расширенная v3", fmt(V3["fires"]), fmt(V3["slices"]), K["years"], "время прихода по VIIRS/MODIS (часы), шаг 12 ч, 500 м + 1 км",
        "основная для следующего обучения"]],
      ["Часть", "Пожары", "Срезы", "Годы", "Разметка", "Назначение"], widths=[3.6, 1.6, 1.6, 1.8, 5.2, 3.6])

doc.add_heading("Источники данных", level=1)
table([["Экспертные контуры пожаров", "44 полигона, атрибуты: время первой термоточки, координата, сенсор, снимок", "10–30 м", "2014–2023", "burned_area (заповедники)"],
       ["NASA FIRMS (архив SP)", "термоточки MODIS C6.1, VIIRS S-NPP, NOAA-20, NOAA-21", "375 м / 1 км", "2012–2024", "API по ключу MAP_KEY"],
       ["MCD64A1 C6.1", "суточная дата выгорания MODIS", "500 м", "2014–2023", "Microsoft Planetary Computer"],
       ["Global Fire Atlas v2", "отдельные пожары: очаг, даты, площадь, суточный растр дат", "500 м", "2002–2024", "Zenodo 14065246, CC BY 4.0"],
       ["ERA5 (Open-Meteo)", "почасовая погода: T, RH, ветер, порывы, осадки, VPD, влажность почвы", "0,25°, 1 ч", "2014–2024", "без ключа"],
       ["GFS (Open-Meteo)", "архив прогнозов (для проверки «на прогнозе»)", "0,25°, 1 ч", "с 2021", "без ключа"],
       ["Sentinel-2 / Landsat (GEE)", "медианные композиты 45 сут до и после пожара", "10–30 м", "2014–2023", "ext_gee из статьи"],
       ["MODIS MOD13A1", "NDVI, 16-суточные композиты", "500 м", "2012–2024", "Planetary Computer"],
       ["ESA WorldCover 2021", "растительный покров → доли классов, топливо FBFM40", "10 м", "2021", "открытый S3"],
       ["Copernicus DEM GLO-90", "высота, уклон, экспозиция", "90 м", "—", "AWS / Planetary Computer"],
       ["NOAA ISD (v3)", "наблюдения 176 метеостанций Казахстана, 8 сроков/сут", "станции", "2013–2024", "без ключа"],
       ["GFS / ECMWF Previous Runs (v3)", "прогнозы с заблаговременностью 1–4 сут", "0,25°, 1 ч", "2024", "Open-Meteo, без ключа"],
       ["Landsat 8/9 (v3)", "топливо до пожара для кубов заповедников", "30 м", "2014–2023", "Planetary Computer"],
       ["OpenStreetMap (v3)", "населённые пункты и дороги (заместители тушения)", "вектор", "2026", "Geofabrik"]],
      ["Источник", "Что используется", "Разрешение", "Период", "Доступ"], widths=[3.4, 6.4, 1.9, 1.9, 3.8], size=8.5)

# ---------- часть A
doc.add_heading("Часть A. Экспертные пожары заповедников", level=1)
sites = ev.site.value_counts()
para(f"Основа — {len(ev)} контуров пожаров, размеченных экспертами: Алтын-Дала {sites.get('AltynDala', 0)}, Коргалжын {sites.get('Korgalzhyn', 0)}, "
     f"Наурзум {sites.get('Naurzum', 0)}, Миргород {sites.get('Mirgorod', 0)}. Для каждого пожара собраны четыре уровня данных.")
doc.add_heading("Событие и погода", level=2)
bullets([f"**Очаг и время возгорания** — координата и время первой термоточки из атрибутов полигона; проверка: очаг не дальше 5 км от контура "
         f"(у {int((ev.ign_to_poly_km == 0).sum()) if 'ign_to_poly_km' in ev else 39} из 44 — внутри контура).",
         "**Геометрия распространения** — дальность от очага, азимуты на дальнюю точку и на центроид гари, площадь, форма, нижняя оценка скорости.",
         "**Погода ERA5** — почасовые ряды 17 переменных с весны до даты снимка; **канадская система FWI** (FFMC, DMC, DC, ISI, BUI, FWI) "
         "по собственной реализации; 67 погодных признаков на событие: условия в час возгорания, средние и экстремумы за 24/48/72 ч, "
         "антецедентные осадки 7–60 суток, устойчивость ветра.",
         "**Разбиения** для честной проверки: по годам (обучение ≤ 2019 — 25 пожаров, проверка 2020 — 6, тест ≥ 2021 — 13) и по заповедникам."])
doc.add_heading("Растровые кубы 30 м и таблица для ML", level=2)
para("Вокруг каждого пожара вырезано окно «контур + 3 км» на сетке 30 м в UTM: всего 77,1 млн пикселей, 17 каналов.")
table([["Состояние топлива до пожара", "6 каналов Sentinel-2/Landsat + NDVI, NDMI, NBR", "21 пожар (годный композит)"],
       ["Рельеф", "высота, уклон, синус и косинус экспозиции", "44"],
       ["Растительный покров", "класс ESA WorldCover", "44"],
       ["Прошлые гари", "сутки с предыдущего пожара в пикселе (инвентарь заповедника)", "31 окно"],
       ["Геометрия", "расстояние до очага; смещение вдоль/поперёк ветра", "44"],
       ["Цель и маски", "контур пожара; зоны других пожаров (исключаются)", "44"],
       ["Контроль разметки", "NBR и dNBR после пожара (не вход модели)", "28 / 20"]],
      ["Группа", "Каналы", "Покрытие"], widths=[4.2, 9.2, 3.6])
para(f"Из кубов собрана таблица для классических ML-моделей: **431 980 пикселей × {len(pix_cols) if isinstance(pix_cols, list) else 90} колонок** "
     "(19 пиксельных + 60 погодных признаков), сбалансированная выборка горевших и негоревших пикселей с весами. Площадь, скорость и азимут "
     "гари в признаки не входят — это цель.")
figure(F("fig2_events_overview.png"), "Рис. 1. Экспертные контуры, очаги и ветер по заповедникам.", 16.5)
figure(F("fig3_cube_example.png"), "Рис. 2. Каналы растрового куба 30 м на примере пожара Naurzum_2019_07_20.", 17.5)

doc.add_heading("Динамика фронта и срезы 24/48/72 ч", level=2)
para(f"Чтобы учить прогноз «через сутки», нужен не только финальный контур, но и время, когда огонь дошёл до каждой точки. "
     f"Для этого скачан архив термоточек FIRMS: **{fmt(fs.n_in_poly.sum())} точек внутри контуров**, точки есть у {int((fs.n_in_poly > 0).sum())} из 44 пожаров, "
     f"медиана {fs.n_overpasses.median():.0f} пролётов на пожар. У {int((fs.first_h.abs() <= 1).sum())} пожаров первая термоточка совпала "
     "со временем возгорания в разметке с точностью до часа.")
bullets(["**Время прихода огня** в каждом пикселе 30 м внутри контура: по ближайшим термоточкам (основной источник, медиана 70 % площади), "
         "по суточной дате MCD64A1 (если термоточек рядом нет) или по дальней термоточке (низкая надёжность, помечено).",
         f"**Динамика**: медианное время выгорания половины площади — {ar.t50_h.median():.1f} ч; {int((ar.burned_by_24h >= .95).sum())} пожаров "
         f"догорели за сутки, {int((ar.t90_h > 72).sum())} горели дольше трёх суток.",
         f"**Срезы**: {len(si1)} срезов по {si1.event.nunique()} пожарам — состояние на момент t (выгоревшее, термоточки, топливо, рельеф, NDVI) и цели — "
         "доля ячейки 500 м, которая выгорит за 24, 48 и 72 ч; к каждому срезу — погода ERA5 и (с 2021 г.) прогноз GFS.",
         "Для крупных пожаров, не помещающихся в сцену 48 × 48 км, те же срезы сохранены на полной сетке пожара."])
figure(F("fig4_arrival_examples.png"), "Рис. 3. Время прихода огня внутри экспертных контуров; точки — термоточки FIRMS, звезда — очаг.", 17.5)
figure(F("fig5_slice_example.png"), "Рис. 4. Срез 500 м: входные каналы и цели на 24/48/72 ч (экспертная часть).", 17.5)

doc.add_heading("Контроль качества: данные физически осмысленны", level=2)
table([["Ветер задаёт направление", "67 % пожаров распространились по ветру (V-тест p = 0,006)"],
       ["Ветер и скорость", "скорость ветра за сутки и скорость распространения: ρ = 0,33 (p = 0,04)"],
       ["Сухость воздуха и рост", "суточный прирост площади растёт с VPD (ρ = +0,21, p = 0,001) и FWI (ρ = +0,15)"],
       ["Прошлые гари — барьер", "выгорает 9 % пикселей гарей до года, 17 % — 1–2 года, 32 % — без гари"],
       ["Влажность топлива", "сухие по NDMI пиксели горят чаще (AUC 0,65)"],
       ["Разметка и снимки согласованы", "медиана dNBR внутри контура 0,137, снаружи 0,034"],
       ["MCD64A1 пропускает степные пожары", f"видит в среднем {ar.mcd64_detect_frac.median() * 100:.0f} % площади контура; у {int((ar.mcd64_detect_frac < .1).sum())} пожаров < 10 %"],
       ["ERA5 и прогноз GFS", "ветер r = 0,96 (GFS сильнее на 0,9 м/с), температура r = 0,99, осадки r = 0,22"]],
      ["Проверка", "Результат"], widths=[5.0, 12.0])
figure(F("fig1_wind_vs_spread.png"), "Рис. 5. Направление распространения относительно ветра и связь ветра со скоростью.", 16)
figure(F("fig6_growth_vs_weather.png"), "Рис. 6. Суточный прирост площади и погода по срезам.", 16.5)

# ---------- часть B
doc.add_heading("Часть B. Расширенная выборка: пожары Казахстана", level=1)
para("44 экспертных пожара достаточно для калибровки физической модели и проверки, но мало для обучения нейросети с нуля. "
     "Поэтому по той же схеме, что в прошлом эксперименте, добавлены все пожары Казахстана из открытого **Global Fire Atlas v2** "
     "(Andela et al., 2019): для каждого пожара известны очаг, даты и суточный растр выгорания по MCD64A1.")
bullets(["**Отбор**: пожары внутри границы Казахстана, длительность ≥ 2 суток и площадь ≥ 1 км²; ячейки пожара — связная область вокруг очага "
         "с датами выгорания в пределах пожара.",
         "**Сетка**: единая национальная сетка 500 м в EPSG:6933 (8 095 × 2 779 ячеек), совпадает с сеткой экспертной части.",
         f"**Термоточки**: {fmt(K['firms_points'])} точек FIRMS по Казахстану за 2012–2024 (март–ноябрь).",
         "**Срез** — конец суток d: состояние = ячейки пожара с датой ≤ d; цели — ячейки с датой в (d, d + 1/2/3 сут]; "
         "до 10 срезов на пожар; рост соседних пожаров в цель не входит (маска valid).",
         "**Погода ERA5** на каждый срез (окна 24/48/72 ч и 7 суток до момента t); полная для 2021–2024, частичная для 2013–2020."])
figure(F("fig7_kz_map.png"), "Рис. 7. Пожары Global Fire Atlas в Казахстане по частям разбиения и экспертные пожары заповедников.", 17)
bp = K["by_part"]
table([[p[0].upper() + p[1:], fmt(v["fires"]), fmt(v["slices"])] for p, v in bp.items()] + [["Всего", fmt(K["fires"]), fmt(K["slices"])]],
      ["Часть (по годам, пожары не пересекаются)", "Пожары", "Срезы"], widths=[9, 3, 3])
table([[r["year"], fmt(r["fires"]), fmt(r["area"]), fmt(r["slices"])] for r in K["by_year"]],
      ["Год", "Пожары", "Выгоревшая площадь, км²", "Срезы"], widths=[2.5, 3, 4.5, 3])
figure(F("fig8_kz_years.png"), "Рис. 8. Пожары и срезы по годам; распределение площади пожаров.", 16.5)
para(f"Типичный пожар — степной (78 % по классификации GFA, ещё 20 % — пашня), медианная площадь {fmt(K['area_median'], 1)} км², "
     f"медианная длительность {K['days_median']:.0f} суток. Новое выгорание за сутки есть в {K['growth_share']['24'] * 100:.0f} % срезов, медианный прирост "
     f"{K['growth_cells_median']['24']:.0f}, {K['growth_cells_median']['48']:.0f} и {K['growth_cells_median']['72']:.0f} ячеек на 24, 48 и 72 ч "
     "(в прошлом отчёте 1, 2 и 2) — класс «новое выгорание» очень редкий, поэтому качество оценивается по F1, а не по точности.")
figure(F("fig9_kz_slice_example.png"), "Рис. 9. Срез расширенной выборки: входные каналы и цели (термоточки показывают, куда пойдёт огонь).", 17.5)
doc.add_heading("Каналы среза (15)", level=2)
table([["Состояние пожара", "выгорело к t; термоточки за сутки (число и FRP); термоточки с начала пожара"],
       ["Топливо", "доли степи, пашни, кустарника/леса, тростника, негорючего (WorldCover); модель топлива FBFM40; NDVI до пожара"],
       ["Прошлые гари", "1 — сгорело ранее в этом или прошлом году, 0,5 — два года назад"],
       ["Рельеф и геометрия", "высота, уклон; расстояние до очага"],
       ["Погода (числа на срез)", "T, RH, VPD, ветер и порывы, компоненты ветра, осадки за 24/48/72 ч; осадки и VPD за 7 сут; влажность почвы"]],
      ["Группа", "Содержание"], widths=[4, 13])

# ---------- часть C (v3)
doc.add_heading("Часть C. Датасет v3: субсуточные метки, станции, прогнозы", level=1)
para("Версия v3 закрывает пробелы, которые в FireSpread_Report_2 названы главными помехами качеству: суточная точность меток, "
     "отсутствие прогноза на 12 ч, погода «из будущего» вместо доступной на момент прогноза, нехватка сведений о растительности и тушении. "
     f"Итог — **{fmt(V3['slices'])} срезов по {fmt(V3['fires'])} пожарам** (моменты прогноза каждые 12 ч, цели 12/24/48/72 ч) и "
     f"{fmt(V3['slices_1km'])} срезов 1 км для {V3['fires_1km']} крупных пожаров.")
table([["Время только с точностью до суток", "время прихода огня в ячейке 500 м по VIIRS 375 м и MODIS, интерполяция внутри пожара; "
        "дата MCD64A1 — только запасной вариант; источник метки сохранён в каждой ячейке"],
       ["MCD64A1 пропускает часть площади", "ячейки с термоточками VIIRS вне контура MCD64A1, связанные с пожаром (+6–18 % площади)"],
       ["Нет прогноза на 12 ч", "моменты прогноза в 00 и 12 UTC (сразу после пролётов VIIRS), цели 12/24/48/72 ч"],
       ["Состояние растительности", "аномалия NDVI к норме 2012–2024, пик NDVI прошлого сезона (сухая трава), лет с прошлой гари; "
        "топливо до пожара по Landsat 8/9 теперь у 44 из 44 экспертных пожаров"],
       ["Погода на момент прогноза", "176 станций NOAA ISD (наблюдения за 24 ч до t); настоящие прогнозы GFS и ECMWF IFS "
        "с заблаговременностью 1–4 сут (Open-Meteo Previous Runs, только 2024 г.)"],
       ["Крупные быстрые пожары", "сцена 96 × 96 ячеек по 1 км для пожаров протяжённостью > 40 км"],
       ["Тушение", "открытых данных нет; заместители по OSM: расстояние до населённого пункта и дороги, плотность дорог"]],
      ["Пробел из отчёта", "Что сделано в v3"], widths=[4.6, 12.4], size=8.5)

bs = V3["by_split"]
table([[n, fmt(bs[k]["fires"]), fmt(bs[k]["slices"])] for k, n in
       [("train", "Обучение (≤ 2021)"), ("val", "Проверка (2022)"), ("test", "Тест (2023–2024)")]]
      + [["Всего", fmt(V3["fires"]), fmt(V3["slices"])], ["Сцены 1 км (крупные пожары)", fmt(V3["fires_1km"]), fmt(V3["slices_1km"])]],
      ["Часть v3", "Пожары", "Срезы"], widths=[9, 3, 3])
para("Каналы среза (23): выгоревшее и время с выгорания; термоточки за 12 и 24 ч, FRP, с начала пожара; прошлые гари и лет с прошлой гари; "
     "FBFM40 и доли WorldCover; высота и уклон; NDVI, аномалия NDVI, пик NDVI прошлого года; расстояние до очага, населённого пункта, дороги; "
     "плотность дорог. Термоточки берутся только до момента t. "
     + " ".join(f"Доля срезов с ростом за {h}: {V3[h]['share_with_growth'] * 100:.0f} %." for h in ("12h", "24h", "72h")).replace("h:", " ч:"))
figure(F("fig11_v3_slice_example.png"), "Рис. 10. Срез v3: состояние, термоточки за 12 ч, цели 12/24/72 ч и источник метки.", 17.5)

doc.add_heading("Проверка меток по экспертным контурам", level=2)
import re
rows = []
names = {"v2 day label": "v2: дата MCD64A1 (полдень)", "v3 src1": "v3: VIIRS*", "v3 src2": "v3: MODIS*", "v3 src3": "v3: интерполяция",
         "v3 src4": "v3: только дата MCD64A1", "v3 src5": "v3: VIIRS вне MCD64A1*"}
for s in V3["label_check_vs_expert"]:
    m = re.match(r"\s*(v2 day label|v3 src\d).*?:\s*([\d.]+) / ([\d.]+) / (\d+)\s*\(bias ([+\-\d.]+)\)", s)
    if m:
        rows.append([names[m.group(1)], fmt(float(m.group(2)), 1), fmt(float(m.group(3)), 1), fmt(int(m.group(4))), m.group(5).replace(".", ",")])
table(rows, ["Метка", "Медиана ошибки, ч", "75-й проц., ч", "Ячеек", "Смещение, ч"], widths=[5.5, 3, 3, 2.5, 3])
para("* экспертное время тоже построено по термоточкам FIRMS, поэтому сравнение не независимое; независимы строки v2 и «только дата MCD64A1». "
     f"Суточная метка v2 ошибается в среднем на ~15 ч и запаздывает. В v3 только {V3['24h']['targets_day_only'] * 100:.0f} % площади целей "
     "имеет время лишь по дате MCD64A1 — этим ячейкам при обучении дан малый вес (для горизонта 12 ч — нулевой).", size=9.5)
figure(F("fig10_v3_label_sources.png"), "Рис. 11. Источники времени прихода огня по годам и ошибка времени против экспертных контуров.", 17)

doc.add_heading("Погода против метеостанций", level=2)
wy = V3["weather_by_year"]
para(f"ERA5 есть у {V3['weather_ok'] * 100:.0f} % срезов v3, станция в пределах 75 км — у {V3['station_ok'] * 100:.0f} %, "
     f"оперативный прогноз (2024) — у {fmt(V3['forecast_ok'] * 100, 1)} %. "
     + ("ERA5 по годам: " + ", ".join(f"{y} — {v * 100:.0f} %" for y, v in wy.items() if v < 0.999) + "; остальные годы — 100 %."
        if any(v < 0.999 for v in wy.values()) else "ERA5 скачан для всех лет 2013–2024."))
src_names = [("era5", "ERA5"), ("gfs", "GFS, прогноз 1–2 сут (2024)"), ("ifs", "ECMWF IFS, прогноз 1–2 сут (2024)")]
table([[n, fmt(QW.loc[(s, "ws_mean"), "n"]), f"{QW.loc[(s, 'ws_mean'), 'bias']:+.2f}".replace(".", ","),
        fmt(QW.loc[(s, "ws_mean"), "mae"], 2), fmt(QW.loc[(s, "ws_mean"), "r"], 2), fmt(QW.loc[(s, "T_max"), "r"], 2),
        f"{QW.loc[(s, 'RH_min'), 'bias']:+.1f}".replace(".", ","), fmt(QW.loc[(s, "RH_min"), "r"], 2)]
       for s, n in src_names if (s, "ws_mean") in QW.index],
      ["Источник", "Срезов", "Ветер: смещ., м/с", "MAE", "r", "T max: r", "RH min: смещ., %", "r"],
      widths=[4.6, 1.6, 2.1, 1.4, 1.3, 1.6, 2.4, 1.3], size=8.5)
para("GFS завышает ветер (+1,4 м/с) и занижает влажность (−6 %); у ECMWF смещение втрое меньше (+0,4 м/с). При обучении на ERA5 и проверке на прогнозе GFS нужна поправка "
     "смещения или обучение на смеси источников.", size=9.5)
figure(F("fig12_v3_weather_vs_station.png"), "Рис. 12. ERA5 и прогнозы 2024 г. против станций: ветер за 24 ч после момента прогноза.", 16)

RS = V3.get("reserves")
if RS:
    doc.add_heading("Экспертные пожары в формате v3 — независимая проверка", level=2)
    para(f"Пожары заповедников пересобраны в тот же формат, что и основная выборка v3: те же каналы ({len(CH3)}), сроки каждые 12 ч, цели 12/24/48/72 ч, "
         f"сетка 500 м и сцены 1 км для крупных пожаров. Отличие — метки: граница берётся из экспертного контура, а не из MCD64A1, "
         f"время прихода — по термоточкам внутри контура. Итог: **срезов — {fmt(RS['slices'])}, пожаров — {RS['events']}**, сцен 1 км — {fmt(RS['slices_1km'])}; "
         f"ERA5 есть у {RS['weather_ok'] * 100:.0f} % срезов, станция — у {RS['station_ok'] * 100:.0f} %.")
    bullets([f"Ячейка относится к пожару, если контур покрывает ≥ 50 % её площади; ячейки, покрытые частично, исключены из оценки. "
             f"Оценка ведётся только внутри окна экспертного куба — за его пределами неизвестно, горело ли.",
             f"Время прихода по термоточкам есть у {RS['24h_targets_timed_by_firms'] * 100:.0f} % площади целей 24 ч, у остальных — дата MCD64A1 "
             "или экстраполяция; веса — те же, что в основной выборке.",
             f"**Пересечение с обучением**: {RS['events_in_gfa_train']} из {RS['events']} пожаров есть и в обучающей части GFA (по MCD64A1). "
             "В индексе указаны их номера (gfa_fire_ids) — при проверке на заповедниках эти пожары нужно исключать из обучения.",
             f"Внутри сцены 48 км — {RS['24h_in_scene_area_share'] * 100:.0f} % прироста за 24 ч. Малые пожары попадают в сцену целиком, "
             "теряется прирост только крупных, для них есть сцены 1 км."])

PM = V3.get("perims")
if PM:
    doc.add_heading("Промежуточные контуры по снимкам — независимая проверка динамики", level=2)
    para(f"Время прихода огня внутри экспертного контура раньше восстанавливалось только по термоточкам. Теперь для {PM['events']} пожаров, "
         f"горевших дольше 12 ч, взяты все снимки Sentinel-2 и Landsat 7/8/9 во время пожара ({PM['snapshots']} безоблачных снимков). По каждому "
         "построена карта «выгорело к моменту снимка»: падение NBR относительно снимка до пожара сравнивается с фоном вне контура на том же снимке, "
         "поэтому сезонное высыхание травы за гарь не принимается.")
    table([["v2: дата MCD64A1 (сутки)", f"{PM['late_share_mcd64_day'] * 100:.0f} %"], ["v3: GFA + VIIRS", f"{PM['late_share_v3_gfa'] * 100:.0f} %"],
           ["Заповедники: термоточки", f"{PM['late_share_expert'] * 100:.0f} %"], ["Заповедники: термоточки + снимки", f"{PM['late_share_expert_img'] * 100:.1f} %".replace(".", ",")]],
          ["Метка", "Доля видимой на снимке гари, которой ещё нет в метке"], widths=[7, 8])
    bullets(["Метки по термоточкам запаздывают в несколько раз меньше суточных меток MCD64A1: снимки независимо подтверждают переход к субсуточным меткам в v3.",
             "Распознавание гари на снимке неполное: в сухой степи и лесостепи часть гари почти не меняет NBR (у Наурзума 2019 распознаётся < 10 %). "
             "Поэтому «на снимке гари нет» учитывается только на надёжных снимках "
             f"({PM['reliable']} из {PM['snapshots']}), где распознано ≥ 70 % пикселей, заведомо сгоревших по термоточкам.",
             "Уточнение: если пиксель уже сгорел на снимке, а метка считает, что позже (> 12 ч), время сдвигается раньше; позже сдвигается только "
             "время без термоточки. Уточнённое время используется в срезах заповедников v3 (код источника 6, вес как у интерполяции)."])
    figure(F("fig13_perims_vs_labels.png"), "Рис. 13. Промежуточные контуры по снимкам против времени прихода по термоточкам и запаздывание меток разных версий.", 17.5)

if V3.get("fwi_ok") is not None or V3.get("forecast_gfsa_by_year"):
    doc.add_heading("Индексы пожарной опасности и архив прогнозов GFS", level=2)
    items = []
    if V3.get("fwi_ok") is not None:
        fm = V3.get("fwi_median_by_growth_24h", {})
        items.append(f"**FWI System (NASA GFWED, MERRA-2 с поправкой осадков, 0,5°)**: FFMC, DMC, DC, ISI, BUI, FWI, DSR на последний полдень до t — "
                     f"у {V3['fwi_ok'] * 100:.0f} % срезов. Медиана FWI: срезы без прироста за 24 ч — {fm.get('no_growth', float('nan')):.0f}, "
                     f"с приростом — {fm.get('growth', float('nan')):.0f}. Свой расчёт, как для событий v1, для 111 тыс. срезов v3 невозможен: погода у них только за 7 суток до t, а DC и DMC требуют ряда с весны. Пропуски (17 %) — зимой и ранней весной: под снегом GFWED индексы не считает (июнь–сентябрь — 98–100 %).")
    if V3.get("forecast_gfsa_by_year"):
        gy = V3["forecast_gfsa_by_year"]
        items.append("**Архив GFS NOAA (AWS, 0,5°)**: прогон за 12 ч до t, заблаговременность 12–84 ч через 6 ч — у "
                     + ", ".join(f"{y} — {v * 100:.0f} %" for y, v in gy.items())
                     + " срезов. Проверка «на прогнозе» теперь возможна на валидационном 2022 и тестовых 2023–2024 гг., а не только на 2024.")
    bullets(items)
    if V3.get("weather_gfsa_vs_stations"):
        g = pd.DataFrame(V3["weather_gfsa_vs_stations"])
        g = g[g["var"] == "ws_mean"]
        nm = {"era5": "ERA5", "gfs_aws": "GFS AWS (прогноз)", "gfs_openmeteo": "GFS Open-Meteo (прогноз)"}
        table([[nm.get(r.source, r.source), r.years, fmt(r.n), f"{r.bias:+.2f}".replace(".", ","), fmt(r.mae, 2), fmt(r.r, 2)] for r in g.itertuples()],
              ["Источник", "Годы", "Срезов", "Ветер: смещ., м/с", "MAE", "r"], widths=[5.5, 2.5, 2, 3, 2, 2])

doc.add_heading("Как использовать при обучении", level=2)
bullets(["Загрузчик scripts/v3_loader.py: load_index(split), load_chunk(file, slices), label_weights(lsrc, Y, horizon) — веса ячеек по надёжности метки.",
         "Признаки, доступные на момент t: каналы X, наблюдения станций st_*_prev24h, era5_*_prev*, индексы gfwed_*, прогнозы fc_gfsa_* (2022–2024) и fc_* (2024). "
         "Поля era5_*_{12..72}h — «идеальный прогноз», только для обучения; st_fut_* — только для проверки.",
         "Крупные пожары: load_index(kind=\"1km\"); там цель — число выгоревших ячеек 500 м (0–4) в ячейке 1 км."])

doc.add_heading("Сравнение с данными прошлого эксперимента", level=1)
table([["Пожары / срезы", "13 738 / 28 590 (2013–2025)", f"{fmt(K['fires'])} / {fmt(K['slices'])} (2013–2024) + 44 / 235 экспертных"],
       ["Граница пожара", "MCD64A1 + термоточки (суточная)", "та же для GFA; экспертный контур 30 м для заповедников"],
       ["Время фронта", "сутки", "сутки (GFA v2); часы по VIIRS/MODIS (v3); часы по термоточкам (заповедники)"],
       ["Горизонты прогноза", "24 / 48 / 72 ч", "24 / 48 / 72 ч; в v3 также 12 ч"],
       ["Погода", "архив прогнозов GFS", "ERA5 (все годы), станции ISD, прогнозы GFS/ECMWF с заблаговременностью (2024)"],
       ["Топливо", "FBFM40", "FBFM40 из WorldCover + доли классов + NDVI MOD13A1"],
       ["Медианный прирост 24/48/72 ч", "1 / 2 / 2 ячейки", f"{K['growth_cells_median']['24']:.0f} / {K['growth_cells_median']['48']:.0f} / {K['growth_cells_median']['72']:.0f} ячейки"]],
      ["Параметр", "FireSpread_Report_2", "Этот датасет"], widths=[4, 5.5, 7.5])

doc.add_heading("Ограничения и флаги качества", level=1)
bullets(["**Суточные метки MCD64A1** в расширенной части: время прихода известно с точностью до суток, а MCD64A1 видит около половины "
         "площади степного пожара. Экспертная часть точнее (часы, контур 30 м) — её стоит использовать для калибровки и итоговой проверки.",
         "**Топливо до пожара**: у экспертных пожаров — Sentinel-2/Landsat (в v3 у 44 из 44); в расширенной части — NDVI MODIS 500 м "
         "и его аномалия; проверенной карты топлива и наземной влажности живой растительности нет.",
         "**Архив прогнозов с заблаговременностью** есть с 2022 г. (GFS AWS) и 2024 г. (ECMWF); на обучающих годах — только ERA5 («идеальный прогноз»).",
         "**Многовременные контуры** есть только по снимкам Sentinel-2/Landsat для 26 экспертных пожаров, и распознавание гари на них неполное; ручной разметки нет.",
         "**Данных о тушении нет** — только заместители доступности по OSM (снимок 2026 г.).",
         "**Погода** — одна точка ERA5 на сцену 48 км; FWI — из GFWED с сеткой 0,5° (MERRA-2), то есть грубее ERA5.",
         "**Флаги экспертной части**: Naurzum_2021_07_22 без термоточек (вероятно, неверная дата, исключён из срезов); у пяти крупных пожаров "
         "половина площади сгорела позже 5 суток — возможно, контур объединяет несколько возгораний.",
         "WorldCover 2021 используется для всех лет; FBFM40 — приближённое сопоставление (LANDFIRE для Казахстана нет)."])

doc.add_heading("Файлы и воспроизведение", level=1)
table([["Таблица событий", "data/events_spread.csv, .gpkg"], ["Погода и FWI", "data/weather_hourly/, weather_daily/, event_weather_features.csv"],
       ["Кубы 30 м", "data/cubes/<пожар>.npz + .json"], ["Таблица для ML", "data/pixel_samples.csv.gz (+ описание колонок)"],
       ["Термоточки, время прихода", "data/firms/, data/arrival/"], ["Срезы экспертной части", "data/slices/, slices_full/, slices_index.csv"],
       ["Срезы Казахстана", "data/ext_kz/slices/<год>.npz, slices_index.csv, events_kz.csv"],
       ["Срезы v3", "data/ext_kz/slices_v3/, slices_1km/, slices_v3_index.csv, slices_v3_channels.json, slices_v3_norm.json"],
       ["Срезы v3 заповедников", "data/ext_kz/slices_v3_reserves/, slices_1km_reserves/, slices_v3_reserves_index.csv"],
       ["Контуры по снимкам", "data/perims/, arrival_img/, qa_perims.*"], ["FWI, прогнозы GFS (AWS)", "data/ext_cache/gfwed/, gfs_aws/; qa_weather_gfsa.csv"],
       ["Проверки v3", "data/ext_kz/qa_labels_v3.*, qa_weather_v3.csv, slices_v3_stats.json"], ["Рисунки", "figures/"]],
      ["Что", "Где (в project_till_nov/task1_dataset)"], widths=[5, 12])
para("Пересборка: скрипты scripts/01–09 (экспертная часть, run_all.sh), 10–16 (Казахстан) и 18–30 (v3, run_v3.sh); Python из burned_area/.venv. "
     "Загрузки кэшируются и повторно не выполняются. Подробный журнал работы — CHECKPOINT_task1_dataset.md и PROGRESS.md.", size=9.5)

doc.add_heading("Ссылки", level=1)
for s in ["Andela N. et al. The Global Fire Atlas of individual fire size, duration, speed and direction. ESSD 11, 529–552 (2019); данные v2: zenodo.org/records/14065246",
          "NASA FIRMS: firms.modaps.eosdis.nasa.gov",
          "MCD64A1 Collection 6.1 User Guide: lpdaac.usgs.gov/documents/1006/MCD64_User_Guide_V61.pdf",
          "ERA5 / GFS через Open-Meteo: open-meteo.com",
          "Field R.D. et al. (2015). Development of a Global Fire Weather Database. NHESS 15, 1407–1423 (GFWED).",
          "NOAA Global Forecast System, архив на AWS: registry.opendata.aws/noaa-gfs-bdp-pds",
          "Microsoft Planetary Computer: planetarycomputer.microsoft.com",
          "ESA WorldCover 2021: esa-worldcover.org; Copernicus DEM GLO-90",
          "Van Wagner C.E. Development and structure of the Canadian Forest Fire Weather Index System (1987)",
          "Scott J.H., Burgan R.E. Standard fire behavior fuel models (FBFM40), RMRS-GTR-153 (2005)"]:
    para("• " + s, size=9, space_after=1)

doc.core_properties.title = "Датасет для прогноза распространения пожаров"
doc.core_properties.subject = "Задание 1: данные ДЗЗ и метеорологии"
doc.save(OUTF)
print("saved", OUTF)
