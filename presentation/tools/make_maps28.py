"""Слайд «Один пример: 28 карт на входе» в стиле презентации: слева список каналов (группа, название, источник), справа все 28 карт
одного примера (пожар 3 апреля 2023, тест) сеткой 7 × 4. Компоновка - как на образце пользователя, оформление - наше (цвета, Arial,
цветовые шкалы карт как на прежнем слайде 9: у каждой группы своя).
  ../.venv/Scripts/python.exe -X utf8 presentation/tools/make_maps28.py      (из project_till_nov)
Выход: presentation/slide9_maps/maps28.png (сетка карт), slide9_28maps.pptx (один слайд, тема из presentation_final2_rev1.pptx).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from prod.config import CH
from prod.engine import FireEngine
from prod.demo import Demo

CASE = 4
P = os.path.join(ROOT, "presentation")
OUT = os.path.join(P, "slide9_maps"); os.makedirs(OUT, exist_ok=True)
BG, INK, BODY, ACC, CARD, LINE, FOOT, ASH = "F7F3EC", "1B1A18", "4F4A43", "C2501E", "FFFDF9", "E4DDD1", "8C8377", "3A3A3A"
GROUPS = {"fire": ("D7263D", "огонь сейчас"), "veg": ("3E9B4F", "растительность"), "prior": ("8E5BB5", "прошлые гари"),
          "relief": ("3E7CB1", "рельеф"), "people": ("E0A100", "люди"), "svc": ("8C8377", "служебные")}
HOT = LinearSegmentedColormap.from_list("hot", ["#F2A541", "#C2501E", "#7A1E05"])
FLAG = ListedColormap(["#E4DDD1", "#4F4A43"])
FIRE = ("burn_age_h", "af_count_12h", "af_frp_12h", "af_count_24h", "af_count_all")
# канал, подпись на карте, строка списка, источник, группа, шкала
MAPS = [("burned_frac", "гарь сейчас", "Доля гари в клетке", "VIIRS, MODIS, GFA", "fire", "ash"),
        ("burn_age_h", "возраст гари", "Возраст гари, ч", "VIIRS, MODIS", "fire", HOT),
        ("af_count_12h", "точки 12 ч", "Термоточки за 12 ч", "FIRMS VIIRS, MODIS", "fire", HOT),
        ("af_frp_12h", "мощность FRP", "Мощность FRP за 12 ч", "FIRMS VIIRS, MODIS", "fire", HOT),
        ("af_count_24h", "точки 24 ч", "Термоточки за 24 ч", "FIRMS VIIRS, MODIS", "fire", HOT),
        ("af_count_all", "точки за пожар", "Термоточки за весь пожар", "FIRMS VIIRS, MODIS", "fire", HOT),
        ("grass_frac", "трава", "Доля травы", "ESA WorldCover", "veg", "YlGn"),
        ("crop_frac", "пашня", "Доля пашни", "ESA WorldCover", "veg", "YlOrBr"),
        ("shrub_tree_frac", "кусты, деревья", "Доля кустов и деревьев", "ESA WorldCover", "veg", "Greens"),
        ("wetland_frac", "болота", "Доля болот", "ESA WorldCover", "veg", "PuBu"),
        ("nonburnable_frac", "негорючее", "Доля негорючего", "ESA WorldCover", "veg", "Greys"),
        ("ndvi", "NDVI", "Зелень NDVI", "MODIS", "veg", "Greens"),
        ("ndvi_anom", "аномалия NDVI", "Аномалия NDVI к норме", "MODIS", "veg", "BrBG"),
        ("ndvi_peak_prev", "пик NDVI", "Пик NDVI прошлого сезона", "MODIS", "veg", "YlGn"),
        ("prior_burn", "горела раньше", "Горела ли клетка раньше", "MCD64A1 с 2001 г.", "prior", "Purples"),
        ("years_since_burn", "лет с гари", "Лет с прошлой гари", "MCD64A1 с 2001 г.", "prior", "Purples_r"),
        ("elev", "высота", "Высота", "Copernicus GLO-90", "relief", "terrain"),
        ("slope_deg", "уклон", "Уклон", "Copernicus GLO-90", "relief", "copper_r"),
        ("dist_settlement_km", "до посёлка", "Расстояние до посёлка", "OpenStreetMap", "people", "Blues_r"),
        ("dist_road_km", "до дороги", "Расстояние до дороги", "OpenStreetMap", "people", "Blues_r"),
        ("road_density", "дороги", "Плотность дорог", "OpenStreetMap", "people", "Oranges"),
        ("dist_ign_km", "до очага", "Расстояние до точки возгорания", "GFA, VIIRS", "people", "Reds_r"),
        ("has_ndvi", "есть NDVI", "Флаг: есть NDVI", "—", "svc", FLAG),
        ("has_ndvi_anom", "есть аномалия", "Флаг: есть аномалия NDVI", "—", "svc", FLAG),
        ("has_ndvi_peak_prev", "есть пик NDVI", "Флаг: есть пик NDVI", "—", "svc", FLAG),
        ("has_elev", "есть высота", "Флаг: есть высота", "—", "svc", FLAG),
        ("has_slope_deg", "есть уклон", "Флаг: есть уклон", "—", "svc", FLAG),
        ("valid", "маска", "Маска области сцены", "—", "svc", FLAG)]
assert len(MAPS) == 28

# ---------------- карты ----------------
E = FireEngine(); Dm = Demo(E)
X = Dm.X[CASE].astype(np.float32); valid = Dm.valid[CASE].astype(bool)
burned = X[CH.index("burned_frac")] > 0.5
rr, cc = np.nonzero(burned); n = 36                                    # 36 × 36 клеток = 18 × 18 км вокруг пожара
r0 = int(np.clip((rr.min() + rr.max()) // 2 - n // 2, 0, 96 - n)); c0 = int(np.clip((cc.min() + cc.max()) // 2 - n // 2, 0, 96 - n))
win = (slice(r0, r0 + n), slice(c0, c0 + n))


def get(k):
    if k == "valid":
        return valid.astype(float)
    if k.startswith("has_"):
        return np.isfinite(X[CH.index(k[4:])]).astype(float)
    return X[CH.index(k)]


GW_IN, GH_IN = 7.05, 5.05                                              # размер сетки на слайде, дюймы
fig, axs = plt.subplots(4, 7, figsize=(GW_IN, GH_IN), facecolor="#" + BG)
fig.subplots_adjust(left=0.004, right=0.996, top=0.955, bottom=0.004, wspace=0.07, hspace=0.30)
for ax, (k, short, _, _, grp, cmap) in zip(axs.flat, MAPS):
    a = get(k)[win]
    if cmap == "ash":
        ax.imshow(np.zeros_like(a), cmap=ListedColormap(["#" + CARD]))
        ax.imshow(np.ma.masked_where(a < 0.5, a), cmap=ListedColormap(["#" + ASH]), interpolation="nearest")
    elif cmap is FLAG:
        ax.imshow(a, cmap=FLAG, vmin=0, vmax=1, interpolation="nearest")
    else:
        if k in FIRE:
            ax.imshow(np.zeros_like(a), cmap=ListedColormap(["#" + CARD]))
            a = np.ma.masked_where(~(a > 0), a)
        else:
            a = np.ma.masked_invalid(a)
        v = a.compressed()
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
        if k == "ndvi_anom":
            m = max(abs(lo), abs(hi), 1e-3); lo, hi = -m, m
        if k in ("crop_frac", "shrub_tree_frac", "wetland_frac", "nonburnable_frac", "grass_frac", "prior_burn"):
            lo, hi = 0, 1                                              # доли - на общей шкале 0..1
        if hi <= lo:
            hi = lo + 1
        ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#" + GROUPS[grp][0]); s.set_linewidth(1.6)
    ax.set_title(short, fontsize=8.6, color="#" + BODY, pad=2.5, fontfamily="Arial")
png = os.path.join(OUT, "maps28.png")
fig.savefig(png, dpi=300, facecolor="#" + BG)
plt.close(fig)

# ---------------- слайд ----------------
prs = Presentation(os.path.join(P, "presentation_final2_rev1.pptx"))
lst = prs.slides._sldIdLst
ids = list(lst)
for k, sid in enumerate(ids):
    if k != 8:                                                          # оставляем слайд 9 «Один пример» - его фон и заголовок
        prs.part.drop_rel(sid.rId); lst.remove(sid)
sl = prs.slides[0]
for sh in list(sl.shapes):
    if not (sh.has_text_frame and sh.text_frame.text.startswith("Один пример")):
        sh._element.getparent().remove(sh._element)
title = [sh for sh in sl.shapes if sh.has_text_frame][0]
rs = title.text_frame.paragraphs[0].runs; rs[0].text = "Один пример: 28 карт на входе"
for x in rs[1:]:
    x.text = ""
if sl.has_notes_slide:
    sl.notes_slide.notes_text_frame.text = (
        "Это все 28 карт, которые получает нейросеть для одного примера: пожар 3 апреля 2023 года из теста, показан участок 18 на 18 "
        "километров вокруг гари. Красная рамка - сам пожар: что уже сгорело, когда, термоточки и их мощность. Зелёная - растительность "
        "по WorldCover и MODIS: доли травы, пашни, кустов, болот и негорючего, NDVI, его аномалия и пик прошлого сезона. Фиолетовая - "
        "прошлые гари, синяя - рельеф, жёлтая - посёлки, дороги и расстояние до очага. Серая - служебные: где данные есть, а где нет. "
        "Кроме карт сеть получает 49 чисел погоды и состояния пожара.")


def rgb(h):
    return RGBColor.from_string(h)


def box(x, y, w, h, fill, line=None):
    s = sl.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill); s.shadow.inherit = False
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    return s


def txt(x, y, w, h, s, size, color, bold=False):
    t = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    t.margin_left = t.margin_right = t.margin_top = t.margin_bottom = 0; t.vertical_anchor = MSO_ANCHOR.MIDDLE; t.word_wrap = False
    p = t.paragraphs[0]; p.alignment = PP_ALIGN.LEFT; r = p.add_run(); r.text = s
    r.font.name = "Arial"; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(color)


# левая карточка: список 28 каналов
X0, Y0, W, H = 0.89, 1.55, 4.95, 5.35
box(X0, Y0, W, H, CARD, LINE)
txt(X0 + 0.22, Y0 + 0.14, W - 0.4, 0.34, "ВХОД: 28 карт × 96 × 96 клеток по 500 м", 13, ACC, bold=True)
row_h = (H - 0.62) / 28
for i, (_, _, name, src, grp, _) in enumerate(MAPS):
    y = Y0 + 0.55 + i * row_h
    box(X0 + 0.22, y + row_h * 0.18, 0.09, row_h * 0.64, GROUPS[grp][0])
    txt(X0 + 0.42, y, 2.6, row_h, name, 9, INK)
    txt(X0 + 3.05, y, 1.75, row_h, src, 8.5, FOOT)

# правая часть: сетка карт + легенда групп + подпись
gx = 5.99
sl.shapes.add_picture(png, Inches(gx), Inches(1.55), Inches(GW_IN), Inches(GH_IN))
lx = gx
for grp, (col, lab) in GROUPS.items():
    box(lx, 6.73, 0.13, 0.13, col)
    txt(lx + 0.19, 6.66, 1.3, 0.27, lab, 10, BODY)
    lx += 0.36 + len(lab) * 0.075
txt(gx, 6.95, GW_IN, 0.25, "Пожар 3 апреля 2023 (тест) · участок 18 × 18 км из сцены 48 × 48 км · + 49 чисел погоды и состояния пожара",
    10, FOOT)
out = os.path.join(OUT, "slide9_28maps.pptx")
prs.save(out)
print("saved", png, out)
