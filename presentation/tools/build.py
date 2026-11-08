"""Редактируемый .pptx из layout.json (замеры браузера), svg.pdf (картинки SVG) и исходников (заметки)."""
import json, re, os, sys, html, io
import pymupdf
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.oxml.ns import qn
from lxml import etree

W, SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PX = 6350                                   # 1920 px -> 13,333 in
FONT = "Arial"
E = lambda v: Emu(int(round(v * PX)))
PT = lambda px: Pt(px * 0.5)

def rgb(s):
    m = re.findall(r"[\d.]+", s or "")
    if len(m) < 3: return None
    if len(m) == 4 and float(m[3]) == 0: return None
    return RGBColor(*(int(float(x)) for x in m[:3]))

def hexof(c): return str(c)

L = json.load(open(os.path.join(W, "layout.json"), encoding="utf-8"))
deck = json.load(open(os.path.join(SRC, "deck.json"), encoding="utf-8"))
svgdoc = pymupdf.open(os.path.join(W, "svg.pdf"))
prs = Presentation(); prs.slide_width = Emu(12192000); prs.slide_height = Emu(6858000)
blank = prs.slide_layouts[6]
ALIGN = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT, "end": PP_ALIGN.RIGHT, "justify": PP_ALIGN.JUSTIFY}

def no_line(shp):
    shp.line.fill.background()

def rect(sl, x, y, w, h, fill, rad=0, line=None, lw=0, name=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rad > 0 else MSO_SHAPE.RECTANGLE
    s = sl.shapes.add_shape(kind, E(x), E(y), E(w), E(h))
    if rad > 0: s.adjustments[0] = min(rad / max(min(w, h), 1), 0.5)
    if fill is not None: s.fill.solid(); s.fill.fore_color.rgb = fill
    else: s.fill.background()
    if line is not None: s.line.color.rgb = line; s.line.width = PT(lw)
    else: no_line(s)
    s.shadow.inherit = False
    if name: s.name = name
    return s

def set_runs(par, runs, ls=0):
    for r in runs:
        if not r["t"]: continue
        run = par.add_run(); run.text = r["t"]; f = run.font
        f.name = FONT; f.size = PT(r["sz"]); f.bold = r["b"]; f.italic = r["i"]
        c = rgb(r["c"]);  f.color.rgb = c if c else RGBColor(0, 0, 0)
        if ls: run._r.get_or_add_rPr().set("spc", str(int(ls * 50)))

def split_lines(runs):
    lines = [[]]
    for r in runs:
        if r["t"] == "\n": lines.append([]); continue
        parts = r["t"].split("\n")
        for k, p in enumerate(parts):
            if k: lines.append([])
            lines[-1].append(dict(r, t=p))
    return lines

def add_text(sl, t):
    x, y, w, h = t["r"]
    ind = 1.2 * t["sz"] if t["li"] else 0
    x -= ind; w += ind
    tb = sl.shapes.add_textbox(E(x), E(y), E(w * 1.03 + 4), E(max(h, t["lh"])))
    tf = tb.text_frame; tf.word_wrap = True; tf.auto_size = None
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for k, line in enumerate(split_lines(t["runs"])):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = ALIGN.get(t["al"], PP_ALIGN.LEFT)
        p.line_spacing = PT(t["lh"])
        if t["li"] and k == 0:
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(ind * PX))); pPr.set("indent", str(-int(ind * PX)))
            bf = etree.SubElement(pPr, qn("a:buFont")); bf.set("typeface", FONT)
            bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", "•")
        set_runs(p, line, t["ls"])
    return tb

def cell_border(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag))
        if tag == "a:lnB" and color is not None:
            ln.set("w", str(int(1 * PX)))
            sf = etree.SubElement(ln, qn("a:solidFill")); c = etree.SubElement(sf, qn("a:srgbClr")); c.set("val", str(color))
        else:
            ln.set("w", "0"); etree.SubElement(ln, qn("a:noFill"))
    # порядок: границы должны идти перед заливкой
    fills = [e for e in tcPr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]
    for f in fills: tcPr.remove(f); tcPr.append(f)

def add_table(sl, T):
    x, y, w, h = T["r"]; rows = T["rows"]; nc = max(len(r["cells"]) for r in rows)
    gf = sl.shapes.add_table(len(rows), nc, E(x), E(y), E(w), E(h)); tb = gf.table
    tblPr = gf._element.graphic.graphicData.tbl.tblPr
    for a in ("firstRow", "bandRow"): tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None: sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"   # «Нет стиля, сетка» -> границы задаём сами
    for j, c in enumerate(rows[0]["cells"]): tb.columns[j].width = E(c["w"])
    for i, r in enumerate(rows):
        tb.rows[i].height = E(r["h"]); bg = rgb(r["bg"])
        for j, c in enumerate(r["cells"]):
            cell = tb.cell(i, j)
            cell.margin_left = cell.margin_right = E(16); cell.margin_top = cell.margin_bottom = E(10)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if bg: cell.fill.solid(); cell.fill.fore_color.rgb = bg
            else: cell.fill.background()
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.alignment = ALIGN.get(c["al"], PP_ALIGN.LEFT)
            set_runs(p, c["runs"])
            cell_border(cell, rgb(c["bb"]))
    gf.name = "Таблица"

def add_chart(sl, r):
    x, y, w, h = r
    cd = CategoryChartData(); cd.categories = ["12 ч", "24 ч", "48 ч", "72 ч"]
    cd.add_series("Наш гибрид", (0.29, 0.40, 0.44, 0.47))
    cd.add_series("Потолок разметки", (0.48, 0.55, 0.59, 0.58))
    gf = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, E(x), E(y), E(w), E(h + 40), cd); ch = gf.chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12); ch.legend.font.name = FONT
    for s, col, bold in zip(ch.series, (RGBColor(0xE2, 0x62, 0x2B), RGBColor(0xCF, 0xC6, 0xB8)), (True, False)):
        s.format.fill.solid(); s.format.fill.fore_color.rgb = col
        s.data_labels.show_value = True; s.data_labels.number_format = "0.00"; s.data_labels.number_format_is_linked = False
        s.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        s.data_labels.font.size = Pt(13); s.data_labels.font.bold = bold; s.data_labels.font.name = FONT
        s.data_labels.font.color.rgb = RGBColor(0xC2, 0x50, 0x1E) if bold else RGBColor(0x6E, 0x66, 0x5B)
    ch.plots[0].gap_width = 60; ch.plots[0].overlap = -8
    va = ch.value_axis; va.visible = False; va.has_major_gridlines = False; va.maximum_scale = 0.7; va.minimum_scale = 0
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(13); ca.tick_labels.font.name = FONT
    ca.format.line.color.rgb = RGBColor(0xBD, 0xB3, 0xA4); ca.has_major_gridlines = False
    gf.name = "График F1 и потолок"

for i, s in enumerate(L):
    sl = prs.slides.add_slide(blank)
    bg = rgb(s["bg"]); sl.background.fill.solid(); sl.background.fill.fore_color.rgb = bg
    for sh in s["shapes"]:
        x, y, w, h = sh["r"]
        fill = rgb(sh["fill"])
        if fill is None and "gradient" in sh["img"]:
            fill = rgb(re.search(r"rgb\([^)]*\)", sh["img"]).group(0))
        full = sh["full"] and sh["bl"] > 0
        if sh.get("kind"):
            k = {"arrow-right": MSO_SHAPE.RIGHT_ARROW, "arrow-left": MSO_SHAPE.LEFT_ARROW, "arrow-down": MSO_SHAPE.DOWN_ARROW,
                 "arrow-up": MSO_SHAPE.UP_ARROW, "chevron": MSO_SHAPE.CHEVRON, "circle": MSO_SHAPE.OVAL}.get(sh["kind"], MSO_SHAPE.RECTANGLE)
            a_ = sl.shapes.add_shape(k, E(x), E(y), E(w), E(h)); a_.fill.solid(); a_.fill.fore_color.rgb = fill; no_line(a_)
            a_.shadow.inherit = False; a_.name = "Стрелка"; continue
        if fill is not None or full:
            rect(sl, x, y, w, h, fill, sh["rad"], rgb(sh["blc"]) if full else None, sh["bl"])
        if sh["bt"] > 0 and not full:
            rect(sl, x, y, w, sh["bt"], rgb(sh["btc"]))
        if sh["bl"] > 0 and not full:
            rect(sl, x, y, sh["bl"], h, rgb(sh["blc"]))
    for r in s["imgs"]:
        x, y, w, h = r
        pix = svgdoc[i].get_pixmap(dpi=192, alpha=True, clip=pymupdf.Rect(x * .75, y * .75, (x + w) * .75, (y + h) * .75))
        pic = sl.shapes.add_picture(io.BytesIO(pix.tobytes("png")), E(x), E(y), E(w), E(h)); pic.name = "Схема"
    for pc in s.get("pics", []):                            # скриншоты: вписать с сохранением пропорций (object-fit: contain)
        x, y, w, h = pc["r"]
        from urllib.parse import urlparse, unquote
        from PIL import Image
        f = unquote(urlparse(pc["src"]).path).lstrip("/")
        iw, ih = Image.open(f).size; k = min(w / iw, h / ih); pw, ph = iw * k, ih * k
        pic = sl.shapes.add_picture(f, E(x + (w - pw) / 2), E(y + (h - ph) / 2), E(pw), E(ph)); pic.name = "Скриншот сайта"
    for T in s["tables"]: add_table(sl, T)
    if s["chart"]: add_chart(sl, s["chart"])
    for t in s["texts"]: add_text(sl, t)
    a = re.search(r"<aside>(.*?)</aside>", open(os.path.join(SRC, "slides", s["id"] + ".html"), encoding="utf-8").read(), re.S)
    if a: sl.notes_slide.notes_text_frame.text = html.unescape(re.sub("<[^>]+>", "", a.group(1)).strip())
prs.core_properties.title = deck["title"]
prs.save(OUT); print("saved", OUT)
