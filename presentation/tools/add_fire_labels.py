"""Подписи «какой пожар» внизу слайдов-примеров в pres_final_rev2.pptx (после вставки слайда 28 карт нумерация: 9, 16, 17, 21, 25).
Повторный запуск не дублирует подпись (ищется по имени фигуры).
  ../.venv/Scripts/python.exe -X utf8 presentation/tools/add_fire_labels.py   (из project_till_nov)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

P = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "presentation", "pres_final_rev2.pptx")
prs = Presentation(P)
T_TEST = "Пожар 3 апреля 2023 · 45,36° с. ш., 75,03° в. д. · тестовая выборка 2023–2024 (модель его не видела)"
T_0906 = ("Пожар 6 сентября 2026 · Костанайская обл., у Кумшика (50,23° с. ш., 64,38° в. д.) · модель его не видела · "
          "факт — гарь по снимку Sentinel-2")
LABELS = {9: (T_TEST, 6.05), 16: (T_0906, 6.55), 17: (T_0906, 6.55), 21: (T_0906, 6.6), 25: (T_TEST, 6.75)}
for n, (t, y) in LABELS.items():
    sl = prs.slides[n - 1]
    for sh in list(sl.shapes):
        if sh.name == "fire_label":
            sh._element.getparent().remove(sh._element)
    tb = sl.shapes.add_textbox(Inches(0.89), Inches(y), Inches(11.5), Inches(0.3)); tb.name = "fire_label"
    tf = tb.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = t
    r.font.name = "Arial"; r.font.size = Pt(11); r.font.color.rgb = RGBColor.from_string("8C8377")
prs.save(P)
print("labels added", list(LABELS))
