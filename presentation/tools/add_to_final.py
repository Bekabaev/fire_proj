"""Добавить слайды из собранной presentation_editable.pptx в presentation_final.pptx (её правит автор вручную) - перед последним слайдом.

  ../../.venv/Scripts/python.exe -X utf8 tools/add_to_final.py site-fireskz site-fireskz-summer     (из папки presentation)

Слайд ищется по id в source/deck.json; фигуры копируются как есть (картинки - с новыми связями), заметки докладчика - тоже.
Номер в подписи «Раздел · N ·» меняется на новую позицию. Перед записью - копия presentation_final_backup.pptx.
"""
import copy, io, json, os, re, shutil, sys
from pptx import Presentation

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
order = json.load(open(os.path.join(HERE, "source", "deck.json"), encoding="utf-8"))["order"]
src = Presentation(os.path.join(HERE, "presentation_editable.pptx"))
fin_path = os.path.join(HERE, "presentation_final.pptx")
shutil.copy(fin_path, os.path.join(HERE, "presentation_final_backup.pptx"))
dst = Presentation(fin_path)
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

for sid in sys.argv[1:]:
    s = src.slides[order.index(sid)]
    d = dst.slides.add_slide(dst.slide_layouts[6])
    for sh in list(d.shapes):                                   # пустой макет без заполнителей
        sh._element.getparent().remove(sh._element)
    bg = s._element.cSld.bg
    if bg is not None:
        d._element.cSld.insert(0, copy.deepcopy(bg))
    for sh in s.shapes:
        el = copy.deepcopy(sh._element)
        for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
            rid = blip.get(R + "embed")
            part = s.part.related_part(rid)
            _, new = d.part.get_or_add_image_part(io.BytesIO(part.blob))
            blip.set(R + "embed", new)
        d.shapes._spTree.append(el)
    if s.has_notes_slide:
        ns = d.notes_slide
        if ns.notes_text_frame is None:                          # в мастере заметок этой презентации нет поля текста -> берём с соседнего слайда
            body = next(sh for sl in dst.slides if sl.has_notes_slide for sh in sl.notes_slide.shapes
                        if sh.is_placeholder and sh.placeholder_format.type == 2)
            ns.shapes._spTree.append(copy.deepcopy(body._element))
        ns.notes_text_frame.text = s.notes_slide.notes_text_frame.text
    # переставить перед последним (итоговым) слайдом
    lst = dst.slides._sldIdLst; ids = list(lst)
    lst.remove(ids[-1]); lst.insert(len(ids) - 2, ids[-1])
    pos = len(ids) - 1
    nums = [int(m.group(1)) for sl in list(dst.slides)[:pos - 1] for sh in sl.shapes if sh.has_text_frame
            for m in [re.search(r"· (\d+) ·", sh.text_frame.text)] if m]
    pos = max(nums) + 1 if nums else pos                        # подписи в final не перенумерованы после удаления слайдов - продолжаем их счёт
    for sh in d.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.text = re.sub(r"^(Результаты · )\d+( ·)", rf"\g<1>{pos}\2", r.text)
    print(sid, "-> слайд", pos)
dst.save(fin_path)
print("готово:", len(dst.slides), "слайдов")
