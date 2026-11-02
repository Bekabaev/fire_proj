# -*- coding: utf-8 -*-
"""Презентация по заданию 1 (датасет) -> project_till_nov/FireSpread_Dataset_Presentation.pptx"""
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT.parent / "FireSpread_Dataset_Presentation.pptx"

DARK = RGBColor(0x2B, 0x21, 0x18)     # обугленный
EMBER = RGBColor(0xC8, 0x45, 0x0E)    # огонь
OCHRE = RGBColor(0xE9, 0xC4, 0x6A)    # сухая степь
SAGE = RGBColor(0x5B, 0x8A, 0x45)     # растительность
TINT = RGBColor(0xF6, 0xF1, 0xEA)     # фон карточек
TINT2 = RGBColor(0xEE, 0xF3, 0xEA)
GREY = RGBColor(0x6B, 0x63, 0x5B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=15, color=DARK, font=BODY, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    """runs: str или список абзацев; абзац = str или список (текст, {опции})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run()
            r.text = t
            r.font.name = o.get("font", font)
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", italic)
            r.font.color.rgb = o.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=15, color=DARK, space=6):
    tb = text(slide, x, y, w, h, items, size=size, color=color)
    for p in tb.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.25))))
        pPr.set("indent", str(-Emu(Inches(0.25))))
        bu = etree.SubElement(pPr, qn("a:buFont"))
        bu.set("typeface", "Arial")
        ch = etree.SubElement(pPr, qn("a:buChar"))
        ch.set("char", "•")
        p.space_after = Pt(space)
    return tb


def title(slide, t, sub=None, color=DARK):
    text(slide, 0.6, 0.38, 12.1, 0.75, t, size=32, font=HEAD, bold=True, color=color,
         anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        text(slide, 0.6, 1.1, 12.1, 0.4, sub, size=15, color=GREY, italic=True)


def card(slide, x, y, w, h, fill=TINT):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def circle(slide, x, y, d, fill, label, size=16):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.name = Pt(size), True, BODY
    r.font.color.rgb = WHITE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return s


def pic(slide, name, x, y, w, h, align="center"):
    path = FIG / name
    iw, ih = Image.open(path).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    px = x + (w - pw) / 2 if align == "center" else x
    py = y + (h - ph) / 2
    return slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))


def caption(slide, x, y, w, t):
    text(slide, x, y, w, 0.35, t, size=11, color=GREY, italic=True)


def table(slide, x, y, w, rows, col_w, size=13, head_fill=DARK, row_h=0.42, hl_rows=()):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    style = tblPr.find(qn("a:tableStyleId"))
    if style is None:
        style = etree.SubElement(tblPr, qn("a:tableStyleId"))
    style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # No Style, Table Grid
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = head_fill
            elif i in hl_rows:
                c.fill.fore_color.rgb = RGBColor(0xFB, 0xE3, 0xD3)
            else:
                c.fill.fore_color.rgb = TINT if i % 2 else WHITE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = ""
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = BODY
            r.font.bold = i == 0 or (j == 0 and nc > 2)
            r.font.color.rgb = WHITE if i == 0 else DARK
    return tbl


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ---------------------------------------------------------------- 1. титул
s = prs.slides.add_slide(BLANK)
bg(s, DARK)
text(s, 0.8, 1.5, 11.5, 0.5, "ПРОЕКТ FIRESPREAD · ЗАДАНИЕ 1", size=16, color=OCHRE, bold=True)
text(s, 0.8, 2.1, 11.5, 2.0,
     "Датасет для ML-моделирования распространения степных пожаров",
     size=44, font=HEAD, bold=True, color=WHITE)
text(s, 0.8, 4.35, 11.0, 0.9,
     "Данные ДЗЗ и метеоданные по пожарам Казахстана 2013–2024 и экспертная разметка заповедников",
     size=20, color=RGBColor(0xE8, 0xDF, 0xD4))
text(s, 0.8, 6.3, 11.5, 0.4, "Октябрь 2026", size=14, color=OCHRE)
notes(s, "Презентация только о датасете (задание 1). Модели (задание 2) здесь не рассматриваются.")

# ---------------------------------------------------------------- 2. состав
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Из чего состоит датасет", "Три части, собранные на одной сетке и в одном формате")
parts = [
    ("1", "Срезы по Казахстану", EMBER, [
        "111 215 срезов, 14 247 пожаров, 2013–2024",
        "сцена 96 × 96 ячеек 500 м (48 × 48 км)",
        "момент прогноза каждые 12 ч, цели 12/24/48/72 ч",
        "+ 3 456 сцен с ячейкой 1 км для 297 крупных пожаров"]),
    ("2", "Экспертные пожары заповедников", SAGE, [
        "44 контура 30 м, 4 заповедника, 2014–2023",
        "кубы 30 м, время прихода огня, погода и FWI",
        "393 среза в том же формате — независимый тест",
        "таблица 431 980 пикселей для классических ML-моделей"]),
    ("3", "Погода и пожарная опасность", DARK, [
        "ERA5 почасово — у 100 % срезов",
        "176 метеостанций NOAA ISD (наблюдения до t)",
        "прогнозы GFS 2022–2024, ECMWF IFS 2024",
        "индексы FWI (GFWED) — у 83 % срезов"]),
]
for k, (num, head, col, items) in enumerate(parts):
    x = 0.6 + k * 4.1
    card(s, x, 1.85, 3.8, 5.1)
    circle(s, x + 0.3, 2.1, 0.75, col, num, size=22)
    text(s, x + 0.3, 3.05, 3.25, 0.9, head, size=19, font=HEAD, bold=True, color=col)
    bullets(s, x + 0.3, 4.0, 3.25, 2.85, items, size=14, space=8)

# ---------------------------------------------------------------- 3. итог в цифрах
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Датасет в цифрах")
stats = [
    ("111 215", "срезов по Казахстану\n(14 247 пожаров, 2013–2024)", EMBER),
    ("393", "срезов экспертных пожаров\nзаповедников — независимый тест", SAGE),
    ("23", "канала в срезе 96 × 96\n(сетка 500 м, EPSG:6933)", DARK),
    ("12 · 24 · 48 · 72 ч", "горизонты прогноза,\nмоменты прогноза каждые 12 ч", EMBER),
    ("176", "метеостанций NOAA ISD\nдля проверки погоды", SAGE),
    ("0,3 ч", "медианная ошибка времени\nприхода огня по VIIRS", DARK),
]
for k, (big, lab, col) in enumerate(stats):
    cx, cy = 0.6 + (k % 3) * 4.1, 1.6 + (k // 3) * 2.75
    card(s, cx, cy, 3.8, 2.45)
    text(s, cx + 0.3, cy + 0.3, 3.3, 0.9, big, size=34 if len(big) < 10 else 28,
         font=HEAD, bold=True, color=col, anchor=MSO_ANCHOR.MIDDLE)
    text(s, cx + 0.3, cy + 1.3, 3.3, 0.9, lab.split("\n"), size=14, color=DARK)

# ---------------------------------------------------------------- 4. источники
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Источники данных", "Только открытые данные; всё скачано и закэшировано, повторная сборка без загрузки")
rows = [
    ["Группа", "Источник", "Что даёт"],
    ["Разметка", "Экспертные контуры заповедников (2014–2023)", "44 пожара: Алтын-Дала 20, Наурзум 11, Коргалжын 10, Миргород 3"],
    ["", "Global Fire Atlas (на основе MCD64A1)", "пожары Казахстана 2013–2024"],
    ["Динамика", "NASA FIRMS: MODIS, VIIRS S-NPP / NOAA-20 / NOAA-21", "время прихода огня, термоточки за 12/24 ч, FRP"],
    ["", "MCD64A1 C6.1, Sentinel-2 / Landsat 7–9", "дата выгорания; промежуточные контуры по снимкам"],
    ["Топливо, рельеф", "ESA WorldCover, MOD13A1 NDVI, Landsat, GLO-90 DEM", "тип топлива (FBFM40), аномалия NDVI, высота, уклон"],
    ["Погода", "ERA5, NOAA ISD (176 станций), GFWED FWI", "реанализ, наблюдения за 24 ч до t, индексы опасности"],
    ["Прогнозы", "GFS (AWS 2022–24), GFS / ECMWF IFS (2024)", "погода, доступная на момент прогноза"],
    ["Тушение", "OpenStreetMap (Geofabrik)", "заместители: до поселения, до дороги, плотность дорог"],
]
table(s, 0.6, 1.75, 12.1, rows, [2.0, 5.0, 5.1], size=14, row_h=0.56)

# ---------------------------------------------------------------- 5. карта Казахстана
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Пожары Казахстана в датасете", "Global Fire Atlas, 2013–2024; звёзды — экспертные пожары заповедников")
pic(s, "pres/kz_map.png", 0.4, 1.55, 8.4, 5.7)
for k, (big, lab, col) in enumerate([
        ("14 247", "пожаров с ростом после момента прогноза (медиана 4,5 км², максимум 7 120 км²)", EMBER),
        ("111 215", "срезов 500 м + 3 456 сцен 1 км для крупных пожаров", DARK),
        ("по годам", "обучение ≤ 2021 · настройка 2022 · тест 2023–2024; пожары между частями не пересекаются", SAGE)]):
    y = 1.75 + k * 1.75
    card(s, 9.0, y, 3.75, 1.55)
    text(s, 9.25, y + 0.15, 3.3, 0.6, big, size=26, font=HEAD, bold=True, color=col)
    text(s, 9.25, y + 0.75, 3.3, 0.75, lab, size=13)

# ---------------------------------------------------------------- 6. срез
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Срез: состояние на t → новое выгорание",
      "Пример: пожар 163 км², сцена 48 × 48 км; цели — что сгорит за 12 / 24 / 48 / 72 ч после t")
pic(s, "pres/slice_example.png", 0.4, 1.55, 12.5, 2.8)
card(s, 0.6, 4.55, 6.0, 2.6)
text(s, 0.85, 4.7, 5.5, 0.4, "23 канала (только информация до t)", size=17, font=HEAD, bold=True, color=EMBER)
text(s, 0.85, 5.2, 5.5, 1.9,
     "выгорело к t, давность горения, термоточки 12/24 ч и FRP, прошлые гари, FBFM40 и доли покрова, "
     "высота, уклон, NDVI, аномалия и пик NDVI, расстояние до очага, поселения, дороги, плотность дорог",
     size=14)
rows = [
    ["Часть", "Годы", "Срезов", "Пожаров"],
    ["обучение", "≤ 2021", "96 087", "12 414"],
    ["настройка", "2022", "7 069", "874"],
    ["тест", "2023–2024", "8 059", "959"],
    ["сцены 1 км", "крупные", "3 456", "297"],
]
table(s, 7.0, 4.55, 5.7, rows, [1.5, 1.4, 1.4, 1.4], size=14, row_h=0.52)

# ---------------------------------------------------------------- 7. факторы
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Какие факторы пожара учтены", "Что влияет на распространение огня и как это представлено в срезах")
rows = [
    ["Фактор", "Как представлен в датасете"],
    ["Тип топлива", "Модель топлива FBFM40 по ESA WorldCover; доли травы, пашни, кустарника, тростника, негорючего"],
    ["Состояние растительности", "NDVI, аномалия NDVI к норме 2012–2024, пик NDVI прошлого сезона (запас сухой травы)"],
    ["Прошлые гари", "Выгоревшие ранее ячейки и число лет с последнего пожара — естественные барьеры"],
    ["Рельеф", "Высота и уклон (Copernicus DEM GLO-90)"],
    ["Погода и опасность", "ERA5, станции за 24 ч до t, прогнозы GFS/ECMWF, индексы FWI: FFMC, DMC, DC, ISI, BUI, FWI"],
    ["Активность огня", "Термоточки VIIRS/MODIS за 12 и 24 ч до t, мощность излучения FRP"],
    ["Доступность для тушения", "Расстояние до населённого пункта и дороги, плотность дорог (OpenStreetMap)"],
]
table(s, 0.6, 1.75, 12.1, rows, [3.4, 8.7], size=14, row_h=0.62)

# ---------------------------------------------------------------- 8. метки времени
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Время прихода огня: точность до часов", "Время в каждой ячейке 500 м по термоточкам; источник метки хранится в ячейке")
pic(s, "pres/label_sources.png", 0.6, 1.6, 6.2, 3.3)
rows = [
    ["Источник метки", "Медиана, ч", "75-й проц., ч"],
    ["VIIRS (375 м)", "0,3", "2,5"],
    ["интерполяция", "0,0", "2,1"],
    ["VIIRS вне MCD64A1", "0,6", "4,1"],
    ["только дата MCD64A1", "135", "158"],
    ["для сравнения: суточная дата MCD64A1", "14,7", "35,1"],
]
table(s, 7.2, 1.75, 5.5, rows, [3.1, 1.2, 1.2], size=13, row_h=0.48, hl_rows=(4, 5))
caption(s, 7.2, 4.7, 5.5, "Ошибка времени против экспертных пожаров заповедников")
bullets(s, 0.6, 5.2, 12.1, 2.0, [
    "Доля площади целей: термоточки 45 %, интерполяция ≈ 30 %, VIIRS вне MCD64A1 10 %, только дата MCD64A1 15 %",
    "Термоточки VIIRS вне контура MCD64A1 добавляют 6–18 % площади пожаров (точность 0,94)",
    "Ячейки «только дата MCD64A1» запаздывают на дни → при обучении малый вес, для горизонта 12 ч — вес 0",
], size=15)

# ---------------------------------------------------------------- 9. снимки
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Проверка меток по спутниковым снимкам", "Промежуточные контуры по Sentinel-2 / Landsat: 26 пожаров, 79 безоблачных снимков")
pic(s, "pres/perims.png", 0.6, 1.6, 12.1, 3.3)
bars = [("суточная дата MCD64A1", 26, EMBER), ("метки по Казахстану", 20, OCHRE),
        ("заповедники: термоточки", 7, SAGE), ("заповедники: + снимки", 3.6, DARK)]
text(s, 0.6, 5.1, 12.1, 0.4, "Доля видимой на снимке гари, которой ещё нет в метке (70 снимков посреди пожара)",
     size=15, bold=True)
for k, (lab, v, col) in enumerate(bars):
    y = 5.6 + k * 0.42
    text(s, 0.6, y, 3.3, 0.36, lab, size=14, anchor=MSO_ANCHOR.MIDDLE)
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.0), Inches(y + 0.04), Inches(7.2 * v / 26), Inches(0.28))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background(); b.shadow.inherit = False
    text(s, 4.1 + 7.2 * v / 26, y, 1.0, 0.36, f"{v:g} %".replace(".", ","), size=14, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 10. погода
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Погода: реанализ, станции и настоящие прогнозы", "Проверка по станциям за 24 ч после момента прогноза")
pic(s, "fig12_v3_weather_vs_station.png", 0.4, 1.55, 12.5, 3.0)
rows = [
    ["Источник", "Ветер: смещение, м/с", "Ветер: MAE", "RH min: смещение, %"],
    ["ERA5 (66 тыс. срезов)", "−0,04", "1,09", "−1,4"],
    ["GFS, прогноз 1–2 сут (2024)", "+1,44", "1,71", "−6,2"],
    ["ECMWF IFS, прогноз 1–2 сут (2024)", "+0,43", "1,21", "−1,5"],
]
table(s, 0.6, 4.75, 7.6, rows, [3.4, 1.6, 1.0, 1.6], size=13, row_h=0.45, hl_rows=(2,))
bullets(s, 8.6, 4.75, 4.15, 2.5, [
    "ERA5 у 100 % срезов, станция ≤ 75 км у 59 %",
    "Прогноз GFS — у всех срезов 2022–2024",
    "GFS завышает ветер и сушит воздух → поправка смещения",
], size=14)

# ---------------------------------------------------------------- 11. экспертные пожары
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Экспертные пожары заповедников", "44 контура 30 м с точкой и временем возгорания, 2014–2023")
pic(s, "fig2_events_overview.png", 0.6, 1.7, 12.1, 3.3)
bullets(s, 0.6, 5.25, 5.9, 2.0, [
    "Событие: очаг, время, длительность, геометрия, нижняя оценка скорости",
    "Погода ERA5 почасово (17 переменных), индексы FWI, 67 признаков",
], size=15)
bullets(s, 6.8, 5.25, 5.9, 2.0, [
    "Кубы 30 м: топливо до пожара (Landsat/Sentinel-2), рельеф, покров, прошлые гари",
    "Таблица 431 980 пикселей × 90 колонок с весами выборки",
], size=15)

# ---------------------------------------------------------------- 12. время прихода огня в заповедниках
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Время прихода огня внутри экспертного контура", "Граница — экспертная (30 м), время — по термоточкам FIRMS, снимкам и MCD64A1")
pic(s, "fig4_arrival_examples.png", 0.6, 1.65, 12.1, 3.9)
for k, (big, lab) in enumerate([("14 987", "термоточек FIRMS внутри контуров"),
                                ("51 %", "контура видит MCD64A1 — пропускает степь"),
                                ("9,9 ч", "медиана t50; 25 пожаров догорели за сутки")]):
    x = 0.6 + k * 4.12
    card(s, x, 5.75, 3.85, 1.35)
    text(s, x + 0.25, 5.85, 3.4, 1.15, [[(big, {"size": 26, "bold": True, "color": EMBER, "font": HEAD})], lab], size=14)

# ---------------------------------------------------------------- 13. физическая осмысленность
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Проверка: данные ведут себя физически", "Контроль качества на 44 экспертных пожарах")
pic(s, "fig1_wind_vs_spread.png", 0.5, 1.7, 7.0, 3.6)
caption(s, 0.6, 5.35, 6.8, "Направление распространения относительно ветра; ветер против скорости")
facts = [
    ("67 %", "пожаров распространились по ветру (V-тест p = 0,006)", EMBER),
    ("9 → 32 %", "доля выгоревшего: недавняя гарь < 1 года против без гари — барьер", SAGE),
    ("0,65", "AUC сухости (NDMI) до пожара: сухие пиксели горят чаще", DARK),
    ("19 → 28", "медиана FWI: срезы без роста за 24 ч против срезов с ростом", EMBER),
]
for k, (big, lab, col) in enumerate(facts):
    y = 1.75 + k * 1.3
    card(s, 7.9, y, 4.85, 1.15)
    text(s, 8.1, y + 0.1, 1.7, 0.95, big, size=22, font=HEAD, bold=True, color=col, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 9.85, y + 0.1, 2.75, 0.95, lab, size=13, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 14. независимый тест
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Экспертные пожары как независимый тест", "Те же срезы 500 м и 23 канала; время прихода уточнено по снимкам")
for k, (big, lab, col) in enumerate([
        ("393", "среза / 35 пожаров\n(+ сцены 1 км)", EMBER),
        ("99 %", "срезов с ERA5,\nFWI у 99,5 %", SAGE),
        ("26 из 35", "пожаров есть среди обучающих —\nпри тесте их исключать", DARK)]):
    x = 0.6 + k * 4.1
    card(s, x, 1.85, 3.8, 2.2)
    text(s, x + 0.3, 2.0, 3.3, 0.8, big, size=34, font=HEAD, bold=True, color=col, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, 2.9, 3.3, 1.0, lab.split("\n"), size=14)
bullets(s, 0.6, 4.4, 12.1, 2.8, [
    "Метки — из экспертного контура (доля ≥ 0,5 в ячейке 500 м) и времени прихода огня; частично покрытые ячейки не учитываются",
    "Пожары Global Fire Atlas внутри окна заменены экспертной разметкой, прочие — в канал прошлых гарей",
    "Без срезов: Naurzum_2021_07_22 (нет термоточек) и 8 пожаров, сгоревших быстрее 12 ч",
    "Отдельный флаг — 5 пожаров с t50 > 5 сут (контур может объединять несколько возгораний)",
], size=15)

# ---------------------------------------------------------------- 15. ограничения
s = prs.slides.add_slide(BLANK)
bg(s, DARK)
text(s, 0.6, 0.45, 12.1, 0.8, "Ограничения данных", size=32, font=HEAD, bold=True, color=WHITE)
left = [
    "15 % площади целей датированы только по MCD64A1 — запаздывают на дни",
    "Архивы прогнозов есть только с 2022 г.; для более ранних лет — ERA5",
    "Распознавание гари по снимкам в сухой степи неполное (у части пожаров < 10 %)",
]
right = [
    "Нет проверенной карты топлива: FBFM40 приближён по WorldCover",
    "Нет наземной влажности живого топлива, данных о тушении, ветра у кромки",
    "OSM — снимок 2026 г., за пределами Казахстана не загружен",
]
for k, (head, items) in enumerate([("Метки и погода", left), ("Нет в открытых данных", right)]):
    x = 0.6 + k * 6.25
    text(s, x, 1.65, 5.8, 0.5, head, size=20, font=HEAD, bold=True, color=OCHRE)
    bullets(s, x, 2.3, 5.8, 3.0, items, size=16, color=RGBColor(0xF2, 0xEC, 0xE4), space=12)
text(s, 0.6, 5.9, 12.1, 1.0,
     "Подробное описание датасета — CHECKPOINT_task1_dataset.md и FireSpread_Dataset_Report.docx",
     size=14, color=OCHRE, italic=True)

prs.save(OUT)
print(OUT)
