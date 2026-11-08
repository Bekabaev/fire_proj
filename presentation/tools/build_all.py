"""Пересборка файлов презентации из исходников слайдов (presentation/source - копия облачной версии).

  ../../.venv/Scripts/python.exe -X utf8 tools/build_all.py        (из папки presentation)

-> presentation.html (все слайды на одной странице), presentation.pdf (печать Edge),
   presentation.pptx (слайд = картинка + заметки докладчика), presentation_editable.pptx (тексты, фигуры, таблицы редактируемые).
Картинки /_blob/<id> облачной версии берутся из site_shots/ по site_shots/blobs.json. Нужен Microsoft Edge.
"""
import json, os, re, subprocess, sys, html, io, pathlib, tempfile
import pymupdf
from pptx import Presentation
from pptx.util import Emu

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # presentation/
SRC = os.path.join(HERE, "source")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PY = sys.executable
WORK = tempfile.mkdtemp(prefix="deck_build_")
BLOBS = json.load(open(os.path.join(HERE, "site_shots", "blobs.json")))


def edge(*args):
    prof = os.path.join(WORK, "edge_prof")
    r = subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}",
                        "--virtual-time-budget=30000", *args], capture_output=True, text=True, timeout=300)
    return r.stdout


deck = json.load(open(os.path.join(SRC, "deck.json"), encoding="utf-8"))
slides = {sid: open(os.path.join(SRC, "slides", sid + ".html"), encoding="utf-8").read() for sid in deck["order"]}

# 1. presentation.html: шапка из прежнего файла (шрифты, печать 1920 x 1080), слайды по порядку
old = open(os.path.join(HERE, "presentation.html"), encoding="utf-8").read()
head = old[:old.index("<body>") + len("<body>")]
rel = lambda h: re.sub(r'src="/_blob/([0-9a-f]+)"', lambda m: f'src="site_shots/{BLOBS[m.group(1)]}"', h)
body = "".join(f'<div class="s">{rel(slides[sid])}</div>' for sid in deck["order"])
open(os.path.join(HERE, "presentation.html"), "w", encoding="utf-8").write(head + body + "</body></html>")
print("html", len(deck["order"]), "слайдов")

# 2. PDF
pdf = os.path.join(HERE, "presentation.pdf")
edge("--no-pdf-header-footer", f"--print-to-pdf={pdf}", pathlib.Path(HERE, "presentation.html").as_uri())
doc = pymupdf.open(pdf); print("pdf", doc.page_count, "стр.")
assert doc.page_count == len(deck["order"]), "число страниц PDF не совпало с числом слайдов"


def notes(sid):
    a = re.search(r"<aside>(.*?)</aside>", slides[sid], re.S)
    return html.unescape(re.sub("<[^>]+>", "", a.group(1)).strip()) if a else ""


# 3. presentation.pptx: слайд = картинка страницы PDF + заметки
prs = Presentation(); prs.slide_width = Emu(12192000); prs.slide_height = Emu(6858000)
for i, sid in enumerate(deck["order"]):
    pg = doc[i]; pix = pg.get_pixmap(matrix=pymupdf.Matrix(1920 / pg.rect.width, 1080 / pg.rect.height))
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    sl.shapes.add_picture(io.BytesIO(pix.tobytes("png")), 0, 0, prs.slide_width, prs.slide_height)
    if notes(sid):
        sl.notes_slide.notes_text_frame.text = notes(sid)
prs.core_properties.title = deck["title"]
prs.save(os.path.join(HERE, "presentation.pptx")); print("pptx (картинки)")

# 4. presentation_editable.pptx: замеры браузера -> фигуры и тексты python-pptx
T = os.path.dirname(os.path.abspath(__file__))
subprocess.run([PY, "-X", "utf8", os.path.join(T, "make_html.py"), SRC, WORK], check=True)
dom = edge("--dump-dom", pathlib.Path(WORK, "extract.html").as_uri())
m = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
assert m, "замеры слайдов не получены"
open(os.path.join(WORK, "layout.json"), "w", encoding="utf-8").write(html.unescape(m.group(1)))
edge("--no-pdf-header-footer", f"--print-to-pdf={os.path.join(WORK, 'svg.pdf')}", pathlib.Path(WORK, "svgonly.html").as_uri())
ed = os.path.join(HERE, "presentation_editable.pptx")
try:
    open(ed, "ab").close()
except PermissionError:                                          # файл открыт в PowerPoint - пишем рядом
    ed = os.path.join(HERE, "presentation_editable_new.pptx"); print("presentation_editable.pptx открыт в PowerPoint -> пишу", ed)
subprocess.run([PY, "-X", "utf8", os.path.join(T, "build.py"), WORK, SRC, ed], check=True)
print("готово; служебные файлы:", WORK)

# 5. speaker_notes.md: заголовок и заметки каждого слайда
md = [f"# {deck['title']} — текст выступающего", ""]
for i, sid in enumerate(deck["order"]):
    h = re.search(r"<h[12][^>]*>(.*?)</h[12]>", slides[sid], re.S)
    md += [f"## {i + 1}. {html.unescape(re.sub('<[^>]+>', '', h.group(1))).strip() if h else sid}", "", notes(sid), ""]
open(os.path.join(HERE, "speaker_notes.md"), "w", encoding="utf-8").write("\n".join(md))
print("speaker_notes.md")
