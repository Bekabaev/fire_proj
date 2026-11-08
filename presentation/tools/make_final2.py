"""presentation_final.pptx -> presentation_final2.pptx: версия для комиссии «ИИ точнее физических моделей, гибрид = ИИ + физика».

  ../.venv/Scripts/python.exe -X utf8 presentation/tools/make_final2_figs.py
  ../.venv/Scripts/python.exe -X utf8 presentation/tools/make_final2.py          (из project_till_nov)

Старые слайды не меняются, кроме: титул (подзаголовок), «Итог» (вывод про ИИ и физику), номера в подписях (по порядку).
Новые: главный вывод, метки (VIIRS/MODIS/MCD64A1), 28 карт, 49 чисел, физические модели, ИИ против физики (график),
ELMFIRE: точность и площадь, ELMFIRE по часам. Числа и источники — в make_final2_figs.py.
"""
import copy, os, re, sys
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "presentation")
FIG = os.path.join(P, "final2_figs")
LIGHT = dict(bg="F7F3EC", ink="1B1A18", body="4F4A43", acc="C2501E", bar="E2622B", card="FFFDF9", line="E4DDD1")
DARK = dict(bg="1B1A18", ink="F5F0E8", body="CFC6B8", acc="F2A541", bar="F2A541", card="26241F", line="3A3833")
FOOT = "8C8377"
M = 0.89                                                       # поле слева, дюймы (как в исходных слайдах)

prs = Presentation(os.path.join(P, "presentation_final.pptx"))
orig = list(prs.slides)
BLANK = next(l for l in prs.slide_layouts if l.name == "Blank")
NOTES_BODY = None
for s in orig:                                                  # тело заметок для новых слайдов (в мастере заметок его нет)
    if s.has_notes_slide and s.notes_slide.notes_placeholder is not None:
        NOTES_BODY = s.notes_slide.notes_placeholder._element; break


def rgb(h):
    return RGBColor.from_string(h)


def text(sl, x, y, w, h, runs, size=15, color=None, bold=False, align=None, anchor=None, th=LIGHT, spacing=None):
    """runs: строка или список абзацев; абзац - строка или список (текст, жирный[, цвет])."""
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if anchor:
        tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        for r in (para if isinstance(para, list) else [(para, bold)]):
            run = p.add_run(); run.text = r[0]; f = run.font
            f.name = "Arial"; f.size = Pt(size); f.bold = r[1]; f.color.rgb = rgb(r[2] if len(r) > 2 else (color or th["body"]))
    return tb


def rect(sl, x, y, w, h, fill, rounded=False, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    if rounded:
        s.adjustments[0] = 0.04
    s.shadow.inherit = False
    return s


def new_slide(title, footer, th=LIGHT, notes=""):
    sl = prs.slides.add_slide(BLANK)
    bg = sl.background.fill; bg.solid(); bg.fore_color.rgb = rgb(th["bg"])
    text(sl, M, 0.89, 11.93, 0.55, title, size=32, bold=True, color=th["ink"], th=th)
    text(sl, M, 6.86, 11.93, 0.2, footer, size=12, color=FOOT, th=th)
    if notes:
        ns = sl.notes_slide
        if ns.notes_placeholder is None and NOTES_BODY is not None:
            ns.shapes._spTree.append(copy.deepcopy(NOTES_BODY))
        ns.notes_placeholder.text_frame.text = notes
    return sl


def table(sl, x, y, w, col_w, rows, th=LIGHT, size=13, head_size=11, row_h=0.42, bold_first=True):
    t = sl.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    tbl = t._tbl; tbl.tblPr.set("firstRow", "0"); tbl.tblPr.set("bandRow", "0")
    sid = tbl.tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tbl.tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"          # No Style, No Grid
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(row_h if i else 0.34)
        for j, v in enumerate(r):
            c = t.cell(i, j); c.margin_left = Inches(0.06 if j else 0); c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.04); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.text = ""
            run = p.add_run(); run.text = str(v); f = run.font; f.name = "Arial"
            if i == 0:
                f.size = Pt(head_size); f.color.rgb = rgb(FOOT)
            else:
                f.size = Pt(size); f.color.rgb = rgb(th["ink"] if (j == 0 and bold_first) else th["body"]); f.bold = bool(j == 0 and bold_first)
            tcPr = c._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag), w="0" if tag != "a:lnB" else "9525")
                if tag == "a:lnB":
                    sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr"), val=th["line"])
                else:
                    etree.SubElement(ln, qn("a:noFill"))
            etree.SubElement(tcPr, qn("a:noFill"))
    return t


def bullets(sl, x, y, w, items, th=LIGHT, size=14, gap=0.06):
    paras = [[("•  ", False, th["acc"])] + (it if isinstance(it, list) else [(it, False)]) for it in items]
    return text(sl, x, y, w, 0.3 * len(items), paras, size=size, th=th, spacing=6)


def pic(sl, f, x, y, w=None, h=None):
    return sl.shapes.add_picture(os.path.join(FIG, f), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


# ======================= новые слайды =======================
N = {}

# --- главный вывод (тёмный, после «Почему трудно») ---
th = DARK
s = new_slide("Главный результат: ИИ точнее физических моделей", "Введение · F1 по всем клеткам новой гари за 72 ч; одна метрика и одни данные для всех моделей; что не доказано — слайд 30", th,
              notes="Главный результат работы — сразу. Мы сравнили модель на основе искусственного интеллекта с тремя физическими моделями "
                    "распространения огня на одних и тех же пожарах и одной метрикой. На тесте 2023–2024 годов, около тысячи пожаров, "
                    "нейросеть точнее канадской физической модели FBP в 2,7 раза, и это статистически надёжно. На независимых экспертных "
                    "контурах заповедников разрыв меньше — в 1,7 раза: спутниковые метки видят только половину гари и сильнее наказывают "
                    "физику за большую площадь. Против симулятора ELMFIRE с той же поправкой скорости гибрид точнее в 2,9 раза на 72 часа; "
                    "ELMFIRE без поправки проигрывает в 8 раз, но это слишком слабый соперник, чтобы на него опираться. Где преимущество не "
                    "доказано — покажу отдельным слайдом. Физику мы не выбрасываем: гибрид берёт у неё направление по ветру.")
for k, (big, small) in enumerate((("×2,7", "U-Net точнее физики FBP на тесте 2023–24 (959 пожаров): F1 0,31 против 0,12; Ротермел — 0,08"),
                                  ("×1,7", "гибрид точнее FBP на контурах экспертов в заповедниках (35 пожаров): F1 0,47 против 0,28"),
                                  ("×2,9", "гибрид точнее ELMFIRE с поправкой скорости на 40 пожарах: F1 0,54 против 0,19 (без поправки — 0,06)"))):
    x = M + k * 3.96
    rect(s, x, 1.82, 3.63, 0.04, th["bar"])
    text(s, x, 2.09, 3.77, 0.83, big, size=60, bold=True, color=th["acc"], th=th)
    text(s, x, 3.06, 3.6, 1.0, small, size=15, th=th)
text(s, M, 4.55, 11.3, 0.9, [[("Физику не выбрасываем. ", True, th["ink"]),
                              ("Гибрид берёт у неё направление по ветру (86 % прогнозов роста идут по ветру, у одной нейросети — 26 %), "
                               "а у ИИ — решение, растёт ли пожар и насколько.", False)]], size=15, th=th)
N["thesis"] = s

# --- метки ---
th = LIGHT
s = new_slide("Метки: когда огонь пришёл в каждую клетку", "Данные · доли по ~227 тыс. новых клеток за 72 ч (обучение + тест, каждый 20-й срез)", th,
              notes="Главная работа по данным — восстановить для каждой клетки время, когда до неё дошёл огонь. Стандартный продукт MCD64A1 "
                    "даёт только день и видит около половины гари. Поэтому мы берём время по термоточкам спутников: VIIRS с разрешением "
                    "375 метров — это 44 процента клеток, причём почти 9 процентов лежат вне контура MCD64A1, то есть это гарь, которую "
                    "стандартный продукт пропускает. Ещё 12 процентов — MODIS, треть — интерполяция между термоточками, и только 13 процентов "
                    "имеют лишь дату. Чем точнее время метки, тем больше её вес в обучении; суточные даты на горизонте 12 часов не учитываются.")
table(s, M, 1.75, 11.56, [4.6, 2.0, 2.3, 2.66], [
    ["ИСТОЧНИК ВРЕМЕНИ ПРИХОДА ОГНЯ", "ДОЛЯ КЛЕТОК", "ТОЧНОСТЬ ВРЕМЕНИ", "ВЕС В ОБУЧЕНИИ"],
    ["Термоточки VIIRS, 375 м (NASA FIRMS)", "43,6 %", "≈ 6 ч", "0,67"],
    ["   из них вне контура MCD64A1", "8,6 %", "", ""],
    ["Интерполяция между термоточками", "32,4 %", "≈ 12 ч", "0,5"],
    ["Только дата MCD64A1 (сутки)", "12,6 %", "≈ 24 ч", "0,33; на 12 ч не учитывается"],
    ["Термоточки MODIS, 1 км", "11,5 %", "≈ 12 ч", "0,5"]], th)
bullets(s, M, 4.55, 11.4, [
    [("Больше половины клеток (55 %) имеют время прихода с точностью до часов ", True, LIGHT["ink"]), ("— по термоточкам VIIRS и MODIS, а не по суточной дате.", False)],
    "8,6 % клеток — гарь, которую стандартный продукт MCD64A1 не видит; без термоточек VIIRS её не было бы в датасете.",
    "Неточная метка весит меньше: сеть не учится на «ступеньках» суточной даты."], th)
N["labels"] = s

# --- 28 карт ---
s = new_slide("28 карт на входе U-Net", "Данные · все карты 96 × 96 клеток по 500 м; при обучении карта поворачивается и отражается вместе с вектором ветра", th,
              notes="Что именно видит нейросеть. 28 карт сцены 48 на 48 километров. Шесть карт описывают сам пожар: какая доля клетки уже "
                    "сгорела, как давно, сколько термоточек и какой они мощности за последние 12 и 24 часа. Восемь — растительность: доли "
                    "травы, пашни, кустов, болот и негорючего по ESA WorldCover, а также индекс зелени NDVI, его отклонение от нормы и пик "
                    "прошлого сезона — это запас сухой травы. Дальше рельеф, прошлые гари и антропогенные факторы: посёлки и дороги, где пожары "
                    "тушат или где есть разрывы. Шесть служебных карт говорят сети, где данных нет.")
table(s, M, 1.7, 11.56, [2.35, 6.66, 2.55], [
    ["ГРУППА", "КАРТЫ", "ИСТОЧНИК"],
    ["Огонь сейчас · 6", "доля сгоревшего в клетке, возраст гари; термоточки за 12 ч (число и мощность FRP), за 24 ч, за весь пожар", "VIIRS, MODIS, Global Fire Atlas"],
    ["Растительность · 8", "доли травы, пашни, кустов и деревьев, болот, негорючего; NDVI, его аномалия, пик NDVI прошлого сезона", "ESA WorldCover, MODIS NDVI"],
    ["Прошлые гари · 2", "горела ли клетка раньше, сколько лет назад", "MCD64A1 (с 2001 г.)"],
    ["Рельеф · 2", "высота, уклон", "Copernicus GLO-90"],
    ["Люди · 4", "расстояние до посёлка и до дороги, плотность дорог, расстояние до точки возгорания", "OpenStreetMap, GFA"],
    ["Служебные · 6", "5 флагов «есть данные» (NDVI ×3, высота, уклон) и маска области", "—"]], th, size=12.5, row_h=0.5)
text(s, M, 5.35, 11.4, 0.6, [[("Карту топлива FBFM40 в сеть не подаём: ", True, LIGHT["ink"]),
                             ("она дублирует доли WorldCover, а с ней физика не стала точнее (проверено на 2022 году).", False)]], size=14, th=th)
N["maps"] = s

# --- 49 чисел ---
s = new_slide("49 чисел: погода и состояние пожара", "Данные · числа входят в сеть через FiLM: в каждом блоке они задают множитель и сдвиг для карт признаков", th,
              notes="Вторая часть входа — 49 чисел. Главное — прогноз погоды на четыре окна: 12, 24, 48 и 72 часа вперёд, по семь величин в "
                    "каждом: максимум температуры, минимум влажности, средний ветер, порывы, ветер по осям u и v и осадки. При обучении это "
                    "реанализ ERA5 с добавленным шумом ошибок прогноза, а на тесте и на сайте — настоящий прогноз GFS, как в работе. Плюс "
                    "погода прошлой недели, канадские индексы пожароопасности FWI из NASA GFWED, сезон и размер пожара. Через механизм FiLM "
                    "погода усиливает или гасит признаки в каждом блоке сети: сушь и ветер — гори, дождь — стоп.")
table(s, M, 1.7, 11.56, [2.75, 5.86, 2.95], [
    ["ГРУППА", "ЧИСЛА", "ИСТОЧНИК"],
    ["Прогноз погоды · 28", "4 окна (0–12, 0–24, 0–48, 0–72 ч) × 7: T max, RH min, средний ветер, порывы, ветер u и v, осадки",
     "обучение: ERA5 + шум ошибок GFS; тест и сайт: прогноз GFS"],
    ["Прошлая погода · 5", "осадки и дефицит влажности VPD за 7 дней, влажность почвы, RH min и ветер за сутки", "ERA5"],
    ["Пожароопасность · 7", "FFMC, DMC, DC, ISI, BUI, FWI, максимум FWI за 3 дня", "NASA GFWED"],
    ["Пожар и сезон · 6", "день года (sin, cos), срок 00 / 12 UTC, полусуток с начала пожара, площадь гари, термоточки за 12 ч", "GFA, VIIRS"],
    ["Пропуски · 3", "нет прогноза / нет прошлой погоды / нет FWI", "—"]], th, size=12.5, row_h=0.52)
text(s, M, 5.15, 11.4, 0.6, [[("Ветер u, v поворачивается вместе с картой ", True, LIGHT["ink"]),
                             ("при аугментации (8 поворотов и отражений) — так сеть учит связь «ветер → направление роста», а не запоминает местность.", False)]],
     size=14, th=th)
N["scalars"] = s

# --- физические модели ---
s = new_slide("Три физические модели, с которыми сравниваем", "Модели · все три подобраны по скорости на 2022 году, как и нейросеть; тест 2023–2024 одинаковый", th,
              notes="С чем мы сравниваем ИИ. Первая — модель Ротермела, 1972 год, лесная служба США. Это классика: скорость фронта из баланса "
                    "тепла по параметрам топлива. На ней построены FARSITE, FlamMap и ELMFIRE. Мы дали ей карту топлива по 40 типам Scott и "
                    "Burgan, посчитанную со спутников. Вторая — канадская FBP, тип O-1b, сухая трава. Она эмпирическая, откалибрована по "
                    "натурным палам и не требует подробной карты топлива; она же работает внутри нашего гибрида. Третья — ELMFIRE, полноценный "
                    "открытый симулятор: Ротермел плюс распространение фронта level-set. Это тот же класс моделей, что моделирование на "
                    "fires.kz. Все три считают скорость по формулам, но не знают, когда пожар погаснет сам или его потушат, — поэтому "
                    "завышают площадь у затухающих пожаров.")
cards = [("Ротермел", "США, 1972 · основа FARSITE, FlamMap, ELMFIRE",
          "Скорость фронта из баланса тепла: 40 типов топлива Scott & Burgan, влажность мёртвого и живого топлива, ветер, уклон.",
          "У нас: карта топлива FBFM40 со спутников (Sentinel-1/2), подбор скорости на 2022 г."),
         ("FBP, тип O-1b", "Канада, 1992 · сухая степная трава",
          "Эмпирическая скорость по натурным палам: ветер, сухость травы (FFMC), зелень (NDVI), уклон. Подробная карта топлива не нужна.",
          "У нас: физика внутри гибрида — эллипс по ветру и время прихода огня по Гюйгенсу."),
         ("ELMFIRE", "Lautenberger, 2013 · открытый симулятор",
          "Полный симулятор: Ротермел + движение фронта level-set по карте топлива и рельефа. Того же класса, что моделирование fires.kz.",
          "У нас: 40 тестовых пожаров, те же входы (топливо, рельеф, прогноз GFS, текущая гарь).")]
for k, (name, sub, body, ours) in enumerate(cards):
    x = M + k * 3.93
    rect(s, x, 1.72, 3.7, 3.3, LIGHT["card"], rounded=True, line=LIGHT["line"])
    text(s, x + 0.25, 1.92, 3.25, 0.4, name, size=20, bold=True, color=LIGHT["ink"], th=th)
    text(s, x + 0.25, 2.36, 3.25, 0.3, sub, size=11.5, color=LIGHT["acc"], th=th)
    text(s, x + 0.25, 2.8, 3.25, 1.3, body, size=13, th=th)
    text(s, x + 0.25, 4.1, 3.25, 0.8, ours, size=12, color=FOOT, th=th)
text(s, M, 5.32, 11.4, 0.7, [[("Общая слабость физики: ", True, LIGHT["ink"]),
                             ("она считает, как быстро идёт огонь, но не знает, погаснет ли пожар сам или его потушат. "
                              "Поэтому все три завышают площадь затухающих пожаров — в 3–5 раз на тесте, ELMFIRE как есть — в 19–24 раза.", False)]], size=14, th=th)
N["phys"] = s

# --- ИИ против физики ---
s = new_slide("ИИ против физики: одна метрика, одни пожары", "Результаты · F1 по всем клеткам новой гари; пороги и скорости подобраны на 2022 году; заповедники — независимая экспертная разметка", th,
              notes="Главный график. Слева тест 2023–2024 годов: восемь тысяч ситуаций, около тысячи пожаров, погода — только прогноз GFS, как "
                    "в реальной работе. Серые и светлые столбцы — физика: Ротермел и FBP, F1 от 0,03 до 0,12. Синие — нейросеть U-Net, от "
                    "0,25 до 0,31, то есть в 2,5–5 раз выше на каждом горизонте. Оранжевые — гибрид, на 1–3 пункта ниже нейросети, но с "
                    "правильным направлением по ветру. Справа независимая проверка: пожары заповедников, контуры нарисованы экспертами. Здесь "
                    "гибрид 0,29–0,47 против 0,11–0,28 у FBP и 0,05–0,19 у Ротермела, а пунктир — потолок, выше которого не прыгнуть из-за "
                    "расхождения спутниковой и экспертной разметки. Гибрид на 72 часа берёт 80 процентов этого потолка.")
pic(s, "ai_vs_phys.png", M, 1.6, w=11.56)
bullets(s, M, 5.75, 11.4, [
    [("Тест: U-Net в 2,7 раза точнее FBP и в 3,7 раза — Ротермела ", True, LIGHT["ink"]), ("(F1 на 72 ч 0,31 против 0,12 и 0,08); 95 % интервалы по пожарам не пересекаются.", False)],
    [("Эксперты: гибрид 0,47 против 0,28 и 0,19 ", True, LIGHT["ink"]), ("— на 12–24 ч разница надёжна, на 48–72 ч интервалы перекрываются (всего 35 пожаров).", False)]], th, size=13.5)
N["ai_vs_phys"] = s

# --- ELMFIRE: точность и площадь ---
s = new_slide("ELMFIRE на 40 пожарах: точность и площадь", "Результаты · ELMFIRE 1.1, сетка 500 м, прогноз GFS, без тушения; ×0,05 — та же поправка скорости, что у нашей FBP", th,
              notes="Подробнее про ELMFIRE. Слева F1 по горизонтам. ELMFIRE как есть — 0,04–0,06: он верно выбирает направление, но при сухой "
                    "траве выжигает всё окно 48 на 48 километров. Если замедлить его в 20 раз — та же поправка, что мы подобрали для своей "
                    "физики, — F1 растёт до 0,19, но всё равно в 2,9 раза ниже гибрида. Справа площадь каждого из 40 пожаров: по горизонтали "
                    "сколько реально сгорело, по вертикали прогноз. Серые точки ELMFIRE лежат на один-два порядка выше диагонали, оранжевые "
                    "точки гибрида — вдоль неё. Физика не умеет решать, будет ли пожар расти; именно это и выучила нейросеть.")
pic(s, "elm_bars.png", M, 1.6, w=6.35)
pic(s, "elm_scatter.png", M + 6.55, 1.6, w=4.8)
bullets(s, M, 5.3, 11.4, [
    [("Как есть ELMFIRE сжигает в 19–24 раза больше факта ", True, LIGHT["ink"]), ("— серые точки на 1–2 порядка выше диагонали.", False)],
    [("Замедленный в 20 раз (×0,05) попадает в масштаб, ", True, LIGHT["ink"]), ("но F1 на 72 ч 0,19 — в 2,9 раза ниже гибрида (0,54).", False)]], th, size=13.5)
N["elm_charts"] = s

# --- ELMFIRE по часам ---
s = new_slide("Как растёт огонь: ELMFIRE и гибрид по часам", "Результаты · пожары теста 3 апреля и 27 июля 2023, окно 48 × 48 км; F1 в подписях: ELMFIRE · ×0,05 · гибрид", th,
              notes="Как это выглядит на карте. Верхний ряд — весенний пожар 3 апреля 2023: реально сгорело 23 квадратных километра. ELMFIRE "
                    "как есть за трое суток сжигает полторы тысячи — серую область. Замедленный ELMFIRE, фиолетовый пунктир, ближе, но уходит "
                    "в сторону. Гибрид, синий контур, почти совпадает с красной реальной гарью: 22 квадратных километра, F1 0,69. Нижний ряд — "
                    "крупный летний пожар 27 июля: реально 208 квадратных километров. ELMFIRE снова выжигает всё вокруг, замедленный почти не "
                    "растёт, а гибрид даёт 373 квадратных километра в правильном направлении, F1 0,70. Интерактивная версия всех 40 пожаров "
                    "с прокруткой времени есть отдельной страницей — могу показать.")
pic(s, "elm_time.png", M, 1.55, h=5.15)
side = M + 9.35
for k, (big, small) in enumerate((("1 584 км²", "ELMFIRE за 72 ч на весеннем пожаре, где сгорело 23 км²"),
                                  ("22 км²", "гибрид на том же пожаре, F1 0,69"),
                                  ("0,70", "F1 гибрида на крупном летнем пожаре (208 км²); ELMFIRE 0,29, ×0,05 — 0,22"))):
    y = 1.75 + k * 1.55
    rect(s, side, y, 2.3, 0.03, LIGHT["bar"])
    text(s, side, y + 0.12, 2.4, 0.5, big, size=28, bold=True, color=LIGHT["acc"], th=th)
    text(s, side, y + 0.68, 2.3, 0.8, small, size=12, th=th)
N["elm_time"] = s


# --- что доказано, а что нет ---
s = new_slide("ИИ против физики: что доказано, а что нет", "Результаты · 95 % интервалы — бутстрэп по пожарам (prod_eval.csv, версии моделей от 04.11); 40 пожаров — сравнение по каждому пожару", th,
              notes="Чтобы не было вопросов, честно: где преимущество ИИ доказано, а где нет. На тесте из почти тысячи пожаров нейросеть "
                    "точнее физики на всех горизонтах, интервалы не пересекаются — это надёжно. На экспертных пожарах заповедников гибрид "
                    "впереди везде, но надёжно — только на 12–24 часа: на 48–72 часа пожаров всего 35, и интервалы перекрываются. Против "
                    "ELMFIRE на 40 пожарах гибрид надёжно лучше на 48–72 часа; на 12 часов разницы нет — за 12 часов большинство пожаров почти "
                    "не растут, и все модели там слабые. Слабое место — крупные летние пожары: в августе 2026 наш основной прогноз занизил их "
                    "в десятки раз, а физический симулятор fires.kz был ближе по масштабу. И две оговорки. Спутниковые метки видят только "
                    "половину гари, а нейросеть училась на таких же метках — это даёт ей фору на тесте; на экспертных контурах разрыв 1,7 "
                    "раза, а не 2,7. Физика у нас упрощённая: сетка 500 метров, карта топлива за одно лето, влажность без суточного хода, "
                    "без переноса искр. Расчёт FARSITE или ELMFIRE с подробными данными может быть точнее.")
table(s, M, 1.65, 11.56, [3.05, 1.05, 2.45, 2.45, 2.56], [
    ["ПРОВЕРКА", "ПОЖАРОВ", "ИИ / ГИБРИД, F1 72 Ч", "ФИЗИКА, F1 72 Ч", "ВЫВОД"],
    ["Тест 2023–24, спутниковые метки", "959", "U-Net 0,31 (0,25–0,34)", "FBP 0,12 (0,10–0,14), Ротермел 0,08", "доказано на 12–72 ч"],
    ["Заповедники, контуры экспертов", "35", "гибрид 0,47 (0,31–0,61)", "FBP 0,28 (0,20–0,36), Ротермел 0,19", "доказано на 12–24 ч; на 48–72 ч — нет"],
    ["40 пожаров сайта", "40", "гибрид 0,54", "ELMFIRE ×0,05: 0,19; как есть: 0,06", "доказано на 48–72 ч; на 12 ч разницы нет"],
    ["Крупные летние пожары, август 2026", "5", "наш основной прогноз 0–11 км²", "fires.kz 56–235 км²; факт 33–316 км²", "физика ближе по масштабу"]],
    th, size=12, row_h=0.56)
bullets(s, M, 4.75, 11.4, [
    [("Метки со спутника видят ~51 % гари, ", True, LIGHT["ink"]), ("а нейросеть училась на таких же метках: на тесте это фора ИИ. На контурах экспертов разрыв меньше — ×1,7 вместо ×2,7.", False)],
    [("Физика у нас упрощённая: ", True, LIGHT["ink"]), ("сетка 500 м, топливо по одному лету 2022, влажность топлива без суточного хода, без переноса искр. Подробный расчёт FARSITE / ELMFIRE может быть точнее.", False)],
    [("Слабое место модели — крупные летние пожары: ", True, LIGHT["ink"]), ("основной прогноз занижает их рост в десятки раз (слайд 34).", False)]], th, size=13)
N["honest"] = s


# ======================= v3: минималистичные слайды сравнения (по скиллам pptx + dataviz) =======================
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION


def col_chart(sl, x, y, w, h, series, vmax):
    """Нативная столбчатая диаграмма: 4 горизонта, серии [(имя, цвет, значения)], подписи значений, без сетки."""
    cd = CategoryChartData(); cd.categories = ["12 ч", "24 ч", "48 ч", "72 ч"]
    for name, _, vals in series:
        cd.add_series(name, vals)
    ch = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name = "Arial"; ch.font.size = Pt(13); ch.font.color.rgb = rgb(LIGHT["body"])
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(13); ch.legend.font.color.rgb = rgb(LIGHT["body"])
    va = ch.value_axis; va.maximum_scale = vmax; va.minimum_scale = 0; va.has_major_gridlines = False; va.visible = False
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(14); ca.tick_labels.font.color.rgb = rgb(LIGHT["ink"])
    ca.format.line.color.rgb = rgb(LIGHT["line"]); ca.has_major_gridlines = False
    plot = ch.plots[0]; plot.gap_width = 70; plot.overlap = -8
    plot.has_data_labels = True; dl = plot.data_labels
    dl.number_format = "0.00"; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(12); dl.font.color.rgb = rgb(LIGHT["ink"])
    for ser, (_, col, _) in zip(plot.series, series):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col); ser.format.line.fill.background()
    return ch


def stat(sl, x, y, w, big, small, th=LIGHT):
    text(sl, x, y, w, 0.95, big, size=60, bold=True, color=th["acc"], th=th)
    text(sl, x, y + 1.05, w, 1.4, small, size=15, th=th)


GREY_L, GREY_D, HYB = "CFC6B8", "8C8377", "C2501E"

# --- гибрид против Ротермела (тест 2023–24) ---
s = new_slide("Гибрид против модели Ротермела", "Результаты · тест 2023–2024: 8 059 ситуаций, 959 пожаров, прогноз погоды GFS; F1 по всем клеткам новой гари", LIGHT,
              notes="Первое сравнение с физикой — модель Ротермела, классика, на которой построены FARSITE и ELMFIRE. Мы дали ей карту "
                    "топлива со спутников и подобрали скорость на 2022 году. На всём тесте, почти тысяча пожаров, она даёт F1 от 0,03 до "
                    "0,08, а наш гибрид — от 0,24 до 0,30. На 72 часа гибрид точнее в 3,6 раза, и это статистически надёжно.")
col_chart(s, M - 0.1, 1.55, 8.0, 4.9, [("Ротермел", GREY_D, [0.034, 0.065, 0.080, 0.083]),
                                     ("Гибрид ИИ + физика", HYB, [0.238, 0.239, 0.276, 0.300])], 0.34)
stat(s, 9.2, 2.1, 3.3, "×3,6", "гибрид точнее Ротермела на 72 ч: F1 0,30 против 0,08")
text(s, 9.2, 4.35, 3.3, 1.2, "Разница надёжна на всех горизонтах: 95 % интервалы по пожарам не пересекаются.", size=13, color=FOOT)
N["roth"] = s

# --- гибрид против ELMFIRE (40 пожаров) ---
s = new_slide("Гибрид против симулятора ELMFIRE", "Результаты · тест: 40 пожаров 2023–2024, сетка 500 м, прогноз GFS; ×0,05 — та же поправка скорости, что у нашей физики", LIGHT,
              notes="Второе сравнение — ELMFIRE, открытый симулятор того же класса, что моделирование fires.kz. Мы запустили его на 40 "
                    "тестовых пожарах с теми же входами. Без настройки он выжигает почти всё окно и даёт F1 около 0,05. Честнее сравнивать "
                    "с замедленным вариантом — с той же поправкой скорости, что у нашей физики: он попадает в масштаб, но на 72 часа "
                    "гибрид всё равно точнее в 2,9 раза. На 12 часов разницы нет — за полсуток большинство пожаров почти не растут.")
col_chart(s, M - 0.1, 1.55, 8.0, 4.9, [("ELMFIRE как есть", GREY_L, [0.041, 0.047, 0.062, 0.064]),
                                     ("ELMFIRE ×0,05", GREY_D, [0.100, 0.153, 0.198, 0.189]),
                                     ("Гибрид ИИ + физика", HYB, [0.160, 0.224, 0.360, 0.542])], 0.62)
stat(s, 9.2, 2.1, 3.3, "×2,9", "гибрид точнее настроенного ELMFIRE на 72 ч: F1 0,54 против 0,19")
text(s, 9.2, 4.35, 3.3, 1.2, "Без настройки ELMFIRE сжигает примерно в 20 раз больше, чем сгорело на самом деле.", size=13, color=FOOT)
N["elm"] = s

# --- один пожар: факт / ELMFIRE / гибрид ---
s = new_slide("Один пожар за 72 часа", "Результаты · пожар 3 апреля 2023, окно 48 × 48 км; красный контур — реальная гарь, чёрное — гарь на старте", LIGHT,
              notes="Как это выглядит на карте. Весенний пожар 3 апреля 2023 года. Слева — что реально сгорело за трое суток: 23 "
                    "квадратных километра. В центре — ELMFIRE без настройки: полторы тысячи квадратных километров, почти всё окно. "
                    "Справа — наш гибрид: 22 квадратных километра, и он лежит там же, где реальная гарь, F1 0,69. Физика без ИИ "
                    "не знает, что этот пожар скоро погаснет; нейросеть знает это по тысячам похожих пожаров.")
pic(s, "fire_case4.png", M, 1.6, w=11.56)
for k, (name, num, sub, col) in enumerate((("Реально сгорело", "23 км²", "за 72 часа", "D7263D"),
                                           ("ELMFIRE", "1 584 км²", "в 70 раз больше факта · F1 0,03", GREY_D),
                                           ("Гибрид", "22 км²", "там же, где факт · F1 0,69", "3E7CB1"))):
    x = M + k * 3.93
    rect(s, x, 5.62, 0.18, 0.18, col)
    text(s, x + 0.3, 5.52, 3.4, 0.4, [[(name + "  ", True, LIGHT["ink"]), (num, True, col)]], size=17, th=LIGHT)
    text(s, x + 0.3, 5.95, 3.4, 0.35, sub, size=13, color=FOOT)
N["fire"] = s

# --- что доказано, а что нет (три карточки) ---
s = new_slide("Что доказано, а что нет", "Результаты · интервалы 95 % — бутстрэп по пожарам; на 40 пожарах — сравнение по каждому пожару", LIGHT,
              notes="Честно о границах результата. Доказано: на тесте из почти тысячи пожаров ИИ точнее физики на всех горизонтах, и "
                    "на 48–72 часа гибрид точнее ELMFIRE. Не доказано: на 35 экспертных пожарах заповедников на 48–72 часа гибрид "
                    "впереди, но пожаров мало, и интервалы перекрываются; на 12 часов против ELMFIRE разницы нет. Оговорки: спутник "
                    "видит около половины гари, а нейросеть училась на таких же метках — это даёт ей фору; физика у нас упрощённая; "
                    "крупные летние пожары модель пока занижает.")
cards = (("+", "Доказано", HYB, ["тест 2023–24, 959 пожаров: ИИ точнее физики на 12–72 ч",
                                 "40 пожаров: гибрид точнее ELMFIRE на 48–72 ч"]),
         ("?", "Не доказано", GREY_D, ["заповедники, 35 пожаров: на 2–3 сутки разница в пределах разброса",
                                        "ELMFIRE на 12 ч: разницы нет"]),
         ("!", "Оговорки", LIGHT["ink"], ["спутник видит ~51 % гари — это фора ИИ на тесте",
                                           "физика упрощённая: сетка 500 м, без суточного хода",
                                           "крупные летние пожары модель занижает"]))
for k, (icon, head, col, items) in enumerate(cards):
    x = M + k * 3.93
    rect(s, x, 1.75, 3.7, 3.95, LIGHT["card"], rounded=True, line=LIGHT["line"])
    circ = s.shapes.add_shape(9, Inches(x + 0.3), Inches(2.05), Inches(0.62), Inches(0.62))     # 9 = овал
    circ.fill.solid(); circ.fill.fore_color.rgb = rgb(col); circ.line.fill.background(); circ.shadow.inherit = False
    tf = circ.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    pp = tf.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER; r = pp.add_run(); r.text = icon
    r.font.name = "Arial"; r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = rgb("FFFFFF")
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    text(s, x + 0.3, 2.9, 3.2, 0.45, head, size=20, bold=True, color=LIGHT["ink"], th=LIGHT)
    text(s, x + 0.3, 3.5, 3.15, 2.4, [[(it, False)] for it in items], size=14, th=LIGHT, spacing=10)
N["honest2"] = s

# ======================= правки старых слайдов =======================
def set_text(shape, new):
    p0 = shape.text_frame.paragraphs[0]
    for p in shape.text_frame.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    runs = p0.runs; runs[0].text = new
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


title = orig[0]
sub = next(sh for sh in title.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Прогноз выгорания"))
set_text(sub, "Прогноз выгорания на 12, 24, 48 и 72 часа: гибрид искусственного интеллекта и физики огня "
              "на тестовых пожарах 2023–2024 точнее физических моделей Ротермела, FBP и ELMFIRE")

fin = orig[-1]
tb = {sh.text_frame.text[:20]: sh for sh in fin.shapes if sh.has_text_frame}
set_text(tb["гибрид «нейросеть + "], "гибрид «ИИ + физика»: направление по ветру (86 %), площадь на тесте ×1,0–1,3")
src = tb["до 80 % от потолка н"]
for k, (txt, h) in enumerate((("ИИ точнее физики: ×2,7 к FBP на тесте (надёжно), ×1,7 на контурах экспертов", 0.65),
                              ("гибрид точнее ELMFIRE на 48–72 ч: F1 0,54 против 0,19 (×0,05) и 0,06 (как есть)", 0.65))):
    el = copy.deepcopy(src._element); fin.shapes._spTree.append(el)
    new = fin.shapes[-1]; new.top = Inches(3.95 + k * 0.68); new.height = Inches(h)
    set_text(new, txt)

nxt = next(v for k, v in tb.items() if k.startswith("обучение на точных"))                                  # «Что дальше»: слабое место
el = copy.deepcopy(nxt._element); fin.shapes._spTree.append(el)
new = fin.shapes[-1]; new.top = Inches(2.92); new.height = Inches(0.65)
set_text(new, "крупные летние пожары: модель занижает их рост — дообучение на них")

o20 = orig[19]                                                    # «Направление и масштаб»: площадь на экспертных пожарах
cap = next(sh for sh in o20.shapes if sh.has_text_frame and sh.text_frame.text.startswith("отношение площади прогноза"))
set_text(cap, "отношение площади прогноза к факту на 12–24 ч; на 48–72 ч — 1,2–1,3; на экспертных пожарах завышена в 1,9–2,0 раза")
for sh in o20.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Негорючие участки"):
        sh.top = Inches(4.75)
    if sh.has_text_frame and sh.text_frame.text.startswith("Направление и масштаб"):
        set_text(sh, "Направление и масштаб на тесте")

# ======================= порядок и номера =======================
o = {i + 1: s for i, s in enumerate(orig)}
order = [o[1], o[2], o[3], N["thesis"], o[4], o[5], o[6], N["labels"], o[7], N["maps"], N["scalars"], o[8], o[9],
         o[10], N["phys"], o[11], o[12], o[13], o[14], o[15], o[16], o[17], o[18],
         o[19], N["roth"], N["elm"], N["fire"], N["honest2"], o[21], o[22], o[26]]
assert len({id(s) for s in order}) == len(order)
SHORT = "--short" in sys.argv                                     # короткая версия: 15 слайдов, только суть
if SHORT:
    order = [o[1], o[2], N["thesis"], o[5], N["maps"], N["scalars"], o[9], o[10], N["phys"], o[17],
             N["roth"], N["elm"], N["fire"], N["honest2"], o[21], o[26]]
lst = prs.slides._sldIdLst; ids = {s.slide_id: el for s, el in zip(prs.slides, list(lst))}
rid = {s.slide_id: el.get(qn("r:id")) for s, el in zip(prs.slides, list(lst))}
want = [ids[s.slide_id] for s in order]
drop = [rid[s.slide_id] for s in prs.slides if s not in order]
for el in list(lst):
    lst.remove(el)
for el in want:
    lst.append(el)
for r in drop:
    prs.part.drop_rel(r)


def replace_all(slide, old, new):
    for sh in slide.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if old in r.text:
                        r.text = r.text.replace(old, new)
    if slide.has_notes_slide and slide.notes_slide.notes_placeholder is not None:
        tf = slide.notes_slide.notes_placeholder.text_frame
        for p in tf.paragraphs:
            for r in p.runs:
                r.text = r.text.replace(old, new)


replace_all(N["thesis"], "слайд 30", f"слайд {order.index(N['honest2']) + 1}")
if o[24] in order:
    replace_all(N["honest"], "(слайд 34)", f"(слайд {order.index(o[24]) + 1})")
else:
    replace_all(N["honest"], "(слайд 34)", "(сравнение с fires.kz, август 2026)")

for pos, s in enumerate(order, 1):
    for sh in s.shapes:
        if not sh.has_text_frame or sh.top < Inches(6.7):
            continue
        r = sh.text_frame.paragraphs[0].runs
        if not r:
            continue
        t = sh.text_frame.text
        m = re.match(r"^([^·]+?) · (\d+(?:\.\d+)?)(.*)$", t, re.S)
        if m:
            rest = m.group(3); rest = " " + rest if rest.startswith("·") else rest
            set_text(sh, f"{m.group(1)} · {pos}{rest}")
        elif re.match(r"^[^·]+ · ", t) and s in N.values():
            mm = re.match(r"^([^·]+?) · (.*)$", t, re.S); set_text(sh, f"{mm.group(1)} · {pos} · {mm.group(2)}")

out = os.path.join(P, "presentation_short.pptx" if SHORT else "presentation_final2.pptx")
prs.save(out); print("готово:", out, len(order), "слайдов")
