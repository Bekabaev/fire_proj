"""Слайд «ELMFIRE против нашей модели» -> presentation/presentation_final.pptx, перед последним («Итог»).

  python elmfire_bench/add_slide.py            (после evaluate.py и make_figs.py)

Оформление берётся с предпоследнего слайда (фон, шрифты, цвета - копии его текстовых полей); номер в подписи
«Результаты · N ·» продолжает предыдущий. Уже вставленный слайд с тем же заголовком заменяется.
Перед записью - копия presentation_final_backup.pptx.
"""
import copy, csv, io, os, re, shutil
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu

HERE = os.path.dirname(os.path.abspath(__file__))
PRES = os.path.join(HERE, "..", "presentation", "presentation_final.pptx")
TITLE = "ELMFIRE против нашей модели: 40 тестовых пожаров"
FG, MUTED, ACC = "F5F0E8", "CFC6B8", "E4572E"

pooled = {}
for line in open(os.path.join(HERE, "results.md"), encoding="utf-8").read().splitlines()[2:]:
    c = [x.strip() for x in line.strip("|").split("|")]
    pooled[(c[0], int(c[2].split()[0]))] = (float(c[3]), float(c[6]))
fmt = lambda x: f"{x:.2f}".replace(".", ",")

shutil.copy(PRES, PRES.replace(".pptx", "_backup.pptx"))
prs = Presentation(PRES)
for i, old in enumerate(list(prs.slides)):                      # повторный запуск - заменить свой слайд
    if any(sh.has_text_frame and sh.text_frame.text == TITLE for sh in old.shapes):
        sid = prs.slides._sldIdLst[i]
        prs.part.drop_rel(sid.rId); prs.slides._sldIdLst.remove(sid)
        buf = io.BytesIO(); prs.save(buf); prs = Presentation(buf)   # иначе новый слайд получит имя части удалённого
        break
ref = prs.slides[len(prs.slides) - 2]
box = {sh.name: sh for sh in ref.shapes}
num = max(int(m.group(1)) for s in prs.slides for sh in s.shapes if sh.has_text_frame
          for m in [re.search(r"Результаты · (\d+) ·", sh.text_frame.text)] if m) + 1

s = prs.slides.add_slide(prs.slide_layouts[6])
for sh in list(s.shapes):
    sh._element.getparent().remove(sh._element)
s._element.cSld.insert(0, copy.deepcopy(ref._element.cSld.bg))


def text(src, x, y, w, h, runs):
    """Копия текстового поля src (стиль первого абзаца/фрагмента) с новыми фрагментами [(текст, цвет, жирный)], '\n' - новый абзац."""
    s.shapes._spTree.append(copy.deepcopy(box[src]._element))
    sh = s.shapes[-1]
    sh.left, sh.top, sh.width, sh.height = Emu(x), Emu(y), Emu(w), Emu(h)
    tf = sh.text_frame
    p0 = tf.paragraphs[0]
    r0 = copy.deepcopy(p0.runs[0]._r)
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    for r in p0.runs:
        p0._p.remove(r._r)
    empty = copy.deepcopy(p0._p)                                 # пустой абзац с тем же оформлением
    p = p0
    for t, col, bold in runs:
        if t == "\n":
            p0._p.getparent().append(copy.deepcopy(empty))
            p = tf.paragraphs[-1]
            continue
        p._p.append(copy.deepcopy(r0))
        run = p.runs[-1]
        run.text, run.font.bold = t, bold
        run.font.color.rgb = RGBColor.from_string(col)


L, W = 812800, 10908792
text("TextBox 11", L, 711200, W, 419100, [(TITLE, FG, True)])
img = os.path.join(HERE, "fig_elmfire_cases.png")
ih = int(W * 690 / 2400)
s.shapes.add_picture(img, Emu(L), Emu(1250000), Emu(W), Emu(ih))
y = 1250000 + ih + 150000
MODELS = (("гибрид", "hybrid"), ("наша FBP", "fbp"), ("ELMFIRE", "elmfire"), ("ELMFIRE ×0,05", "elmfire_adj0.05"))
cols = [[("модель", MUTED, False)] + [(n, ACC if k.startswith("elmfire") else FG, True) for n, k in MODELS],
        [("F1: 12 / 24 / 48 / 72 ч", MUTED, False)] +
        [(" / ".join(fmt(pooled[(k, h)][0]) for h in (12, 24, 48, 72)), FG, k == "hybrid") for _, k in MODELS],
        [("площадь 72 ч", MUTED, False)] + [("×" + f"{pooled[(k, 72)][1]:.1f}".replace(".", ","), FG, False) for _, k in MODELS]]
for (dx, w), col in zip(((0, 1350000), (1350000, 2100000), (3450000, 1050000)), cols):
    runs = []
    for i, r in enumerate(col):
        runs += ([("\n", None, None)] if i else []) + [r]
    text("TextBox 13", L + dx, y, w, 1100000, runs)
a72, h72, f72 = pooled[("elmfire_adj0.05", 72)][0], pooled[("hybrid", 72)][0], pooled[("fbp", 72)][0]
text("TextBox 27", L + 4750000, y, W - 4750000, 1250000, [
    ("ELMFIRE верно выбирает направление по ветру, но как есть ", MUTED, False),
    ("выжигает всё окно 48 км", FG, True),
    (" (у 30 из 40 пожаров площадь в 5+ раз больше факта), а при влажной траве не растёт. "
     "С той же поправкой скорости, что у нашей FBP (×0,05), попадает в масштаб и на 12–24 ч точнее неё, "
     f"но на 72 ч F1 {fmt(a72)} против {fmt(f72)} у FBP и {fmt(h72)} у гибрида — ", MUTED, False),
    ("гибрид выигрывает за счёт решения «растёт ли пожар»", ACC, True), (".", MUTED, False)])
text("TextBox 28", L, 6273800, W, 182880, [(f"Результаты · {num} · ELMFIRE 1.1, сетка 500 м, погода GFS, без тушения; "
                                            "F1 по 40 пожарам вместе; факт - гарь по спутникам", "8C8377", False)])

notes = ("Сравнили с ELMFIRE - открытым физическим симулятором того же класса, что моделирование fires.kz. "
         "Те же 40 тестовых пожаров, те же входы: топливо, рельеф, погода прогноза GFS, старт - текущая гарь. "
         "ELMFIRE хорошо выбирает направление по ветру, но не умеет решать, будет ли пожар расти: при сухой траве и ветре он "
         "выжигает всё окно 48 на 48 километров за двое-трое суток, а при влажной траве не растёт вовсе. Реальные степные пожары "
         "чаще затухают - их тушат, ночью растёт влажность. Поэтому F1 у ELMFIRE 0,04-0,06, у нашей физики 0,06-0,32, "
         "у гибрида 0,16-0,54. Если замедлить ELMFIRE так же, как нашу физику, - в 20 раз, множитель подобран на 2022 годе, - "
         "площадь становится близкой к факту, и на первые сутки он даже точнее нашей физики, но к трём суткам отстаёт: 0,19. "
         "Главное преимущество гибрида - нейросеть, которая решает, растёт ли пожар.")
ns = s.notes_slide
if ns.notes_text_frame is None:
    body = next(sh for sl in prs.slides if sl.has_notes_slide for sh in sl.notes_slide.shapes
                if sh.is_placeholder and sh.placeholder_format.type == 2)
    ns.shapes._spTree.append(copy.deepcopy(body._element))
ns.notes_text_frame.text = notes

lst = prs.slides._sldIdLst; ids = list(lst)                     # перед последним («Итог»)
lst.remove(ids[-1]); lst.insert(len(ids) - 2, ids[-1])
prs.save(PRES)
print(f"слайд {len(prs.slides) - 1} из {len(prs.slides)}, подпись «Результаты · {num}»")
