"""pres_final_rev2.pptx, слайд 24: вместо старого локального ELMFIRE (40 тестовых пожаров) - моделирование fires.kz против гибрида
на 3 пожарах 2026 года, F1 на 72 ч, факт - гарь по снимкам Sentinel-2 (presentation/fireskz_*/numbers_s2.json).
  ../.venv/Scripts/python.exe -X utf8 presentation/tools/slide24_fireskz.py   (из project_till_nov)
"""
import os, json
from pptx import Presentation
from pptx.chart.data import CategoryChartData

P = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "presentation")
FIRES = [("0907", "7.09 · Караганд."), ("0906", "6.09 · Костанай."), ("0919", "19.09 · Абайская")]
kz, hy = [], []
for f, _ in FIRES:
    d = json.load(open(os.path.join(P, f"fireskz_{f}", "numbers_s2.json"), encoding="utf-8"))
    kz.append(round(d["kz_F1_72"], 2)); hy.append(round(d["hyb_F1"]["72h"], 2))
print("fires.kz", kz, "hybrid", hy)

prs = Presentation(os.path.join(P, "pres_final_rev2.pptx"))
s = prs.slides[23]
sh = list(s.shapes)


def txt(i, t):
    tf = sh[i].text_frame; rs = tf.paragraphs[0].runs; rs[0].text = t
    for r in rs[1:]:
        r.text = ""


assert sh[0].text_frame.text.startswith("Гибрид против симулятора ELMFIRE")
txt(0, "Гибрид против моделирования fires.kz")
cd = CategoryChartData(); cd.categories = [n for _, n in FIRES]
cd.add_series("fires.kz", kz); cd.add_series("Гибрид ИИ + физика", hy)
ch = sh[1].chart; ch.replace_data(cd)
ch.value_axis.maximum_scale = 0.5
txt(2, "2 из 3")
txt(3, "пожара гибрид точнее fires.kz; fires.kz завышает площадь в 4–20 раз")
s.notes_slide.notes_text_frame.text = (
    "Второе сравнение — с моделированием сайта fires.kz. Он считает только текущие пожары, поэтому сравнение на трёх пожарах сентября "
    "2026 года, которые наша модель не видела. Факт — гарь по снимкам Sentinel-2. На двух пожарах гибрид точнее: в Карагандинской области "
    "fires.kz увёл огонь в другую сторону, в Костанайской направление угадали оба, но у fires.kz площадь в 6 раз больше. На крупном пожаре "
    "в Абайской области точнее fires.kz: его большой эллипс накрыл почти всю гарь, а наша модель рост занизила. Общее: fires.kz всегда "
    "завышает площадь в 4–20 раз, наша модель ближе по масштабу, но крупные пожары пока занижает. Три пожара — это пример, а не статистика.")
for p_ in s.shapes:
    if p_.name == "fire_label":
        p_._element.getparent().remove(p_._element)
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
tb = s.shapes.add_textbox(Inches(0.89), Inches(6.6), Inches(11.5), Inches(0.3)); tb.name = "fire_label"
r = tb.text_frame.paragraphs[0].add_run()
r.text = "F1 на 72 ч, факт — гарь по снимкам Sentinel-2; 3 пожара сентября 2026, модель их не видела; fires.kz за 2023–2025 моделирование не отдаёт"
r.font.name = "Arial"; r.font.size = Pt(11); r.font.color.rgb = RGBColor.from_string("8C8377")
prs.save(os.path.join(P, "pres_final_rev2.pptx")); print("saved")
