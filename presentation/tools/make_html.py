"""Две служебные страницы из исходников слайдов (шрифт Arial - тот, что будет в .pptx):
extract.html - JS измеряет блоки/тексты/таблицы и пишет JSON в <pre id=out> (для --dump-dom);
svgonly.html - видны только SVG на прозрачном фоне (картинки для .pptx)."""
import json, sys, os, re, pathlib
SRC, OUTD = sys.argv[1], sys.argv[2]
# картинки слайдов (/_blob/<id> в облачной версии) -> локальные файлы presentation/site_shots
SHOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site_shots")
BLOBS = json.load(open(os.path.join(SHOTS, "blobs.json")))
def local(h):
    return re.sub(r'src="/_blob/([0-9a-f]+)"', lambda m: 'src="' + pathlib.Path(SHOTS, BLOBS[m.group(1)]).as_uri() + '"', h)
d = json.load(open(os.path.join(SRC, "deck.json"), encoding="utf-8"))
body = "".join(f'<div class="s" data-id="{sid}">{local(open(os.path.join(SRC, "slides", sid + ".html"), encoding="utf-8").read())}</div>\n' for sid in d["order"])
base = """@page{size:1920px 1080px;margin:0}*{margin:0;padding:0;box-sizing:border-box;font-family:Arial,sans-serif!important}
html,body{background:#555}.s{width:1920px;height:1080px;position:relative;overflow:hidden;break-after:page}
.s>section{position:absolute;inset:0;width:1920px;height:1080px}aside{display:none!important}ul,ol{padding-left:1.2em}li{margin:.15em 0}
table{border-collapse:collapse}th,td{padding:14px 16px;border-bottom:1px solid #D9D0C2;text-align:left}th{font-weight:600;color:#6E665B}"""
JS = r"""
function cs(e){return getComputedStyle(e)}
function R(e,o){const r=e.getBoundingClientRect();return [r.left-o.left,r.top-o.top,r.width,r.height]}
function runs(e){const out=[];const w=document.createTreeWalker(e,NodeFilter.SHOW_TEXT|NodeFilter.SHOW_ELEMENT);let n;
 while(n=w.nextNode()){ if(n.nodeType==1){ if(n.tagName=='BR') out.push({t:'\n'}); continue }
  let t=n.textContent.replace(/\s+/g,' '); if(!t.trim()&&!out.length) continue; const s=cs(n.parentElement);
  if(s.textTransform=='uppercase') t=t.toUpperCase();
  out.push({t:t,b:parseInt(s.fontWeight)>=600,i:s.fontStyle=='italic',c:s.color,sz:parseFloat(s.fontSize)}) }
 if(out.length){out[0].t=out[0].t.replace(/^\s+/,''); out[out.length-1].t=out[out.length-1].t.replace(/\s+$/,'')} return out}
function hasDirectText(e){for(const c of e.childNodes) if(c.nodeType==3&&c.textContent.trim()) return true; return false}
const res=[...document.querySelectorAll('.s')].map(s=>{
 const o=s.getBoundingClientRect(), sec=s.firstElementChild, ss=cs(sec);
 const out={id:s.dataset.id,bg:ss.backgroundColor,shapes:[],texts:[],imgs:[],pics:[],tables:[],chart:null}; const skip=new Set(); const done=new Set();
 sec.querySelectorAll('svg').forEach(v=>{ if((v.getAttribute('aria-label')||'').startsWith('Столбцы')){out.chart=R(v.parentElement,o); [v.parentElement,...v.parentElement.querySelectorAll('*')].forEach(x=>skip.add(x))}
   else {out.imgs.push(R(v,o)); [v,...v.querySelectorAll('*')].forEach(x=>skip.add(x))}});
 sec.querySelectorAll('img').forEach(v=>{out.pics.push({r:R(v,o),src:v.getAttribute('src')}); skip.add(v)});
 sec.querySelectorAll('table').forEach(t=>{ out.tables.push({r:R(t,o),rows:[...t.rows].map(tr=>({bg:cs(tr).backgroundColor,h:R(tr,o)[3],
   cells:[...tr.cells].map(td=>{const c=cs(td);return {w:R(td,o)[2],runs:runs(td),al:c.textAlign,sz:parseFloat(c.fontSize),bb:c.borderBottomColor}})}))});
   [t,...t.querySelectorAll('*')].forEach(x=>skip.add(x))});
 for(const e of sec.querySelectorAll('*')){
  if(skip.has(e)||e.closest('aside')) continue; const st=cs(e), r=R(e,o);
  const bt=parseFloat(st.borderTopWidth), bl=parseFloat(st.borderLeftWidth);
  if(st.backgroundColor!='rgba(0, 0, 0, 0)'||st.backgroundImage!='none'||bt>0||bl>0)
   out.shapes.push({r:r,kind:e.tagName=='X-SHAPE'?e.getAttribute('kind'):null,fill:st.backgroundColor,img:st.backgroundImage,rad:parseFloat(st.borderTopLeftRadius)||0,
     bt:bt,btc:st.borderTopColor,bl:bl,blc:st.borderLeftColor,full:bl>0&&parseFloat(st.borderRightWidth)>0});
  if(done.has(e)) continue;
  if(hasDirectText(e)||(['H1','H2','H3','P','LI'].includes(e.tagName)&&e.textContent.trim())){
   [e,...e.querySelectorAll('*')].forEach(x=>done.add(x));
   const pl=parseFloat(st.paddingLeft),pt=parseFloat(st.paddingTop),pr=parseFloat(st.paddingRight),pb=parseFloat(st.paddingBottom);
   out.texts.push({r:[r[0]+pl,r[1]+pt,r[2]-pl-pr,r[3]-pt-pb],li:e.tagName=='LI',runs:runs(e),al:st.textAlign,lh:parseFloat(st.lineHeight)||parseFloat(st.fontSize)*1.2,
     sz:parseFloat(st.fontSize),ls:parseFloat(st.letterSpacing)||0,
     valign:(cs(e).display=='flex'&&st.alignItems=='center')?'m':'t'})}
 }
 return out});
document.body.innerHTML='<pre id="out"></pre>'; document.getElementById('out').textContent=JSON.stringify(res);
"""
open(os.path.join(OUTD, "extract.html"), "w", encoding="utf-8").write(
    f'<!doctype html><html><head><meta charset="utf-8"><style>{base}</style></head><body>{body}'
    f'<script>document.fonts.ready.then(()=>{{{JS}}})</script></body></html>')
svg = base.replace("html,body{background:#555}", "html,body{background:transparent!important}") + \
    ".s section{background:transparent!important}.s section *{visibility:hidden!important}.s section svg,.s section svg *{visibility:visible!important}"
open(os.path.join(OUTD, "svgonly.html"), "w", encoding="utf-8").write(
    f'<!doctype html><html><head><meta charset="utf-8"><style>{svg}</style></head><body>{body}</body></html>')
print("ok", len(d["order"]))
