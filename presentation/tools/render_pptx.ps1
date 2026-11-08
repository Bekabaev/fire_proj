# Рендер слайдов pptx в PNG через PowerPoint (Windows). Использование:
#   powershell -File render_pptx.ps1 <file.pptx> <out_dir> [slides, напр. "3,5,7"]
param([string]$File, [string]$OutDir, [string]$Only = "")
New-Item -ItemType Directory -Force $OutDir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open((Resolve-Path $File).Path, -1, 0, 0)
$want = @(); if ($Only) { $want = $Only.Split(",") | ForEach-Object { [int]$_ } }
$n = 0
foreach ($sl in $pres.Slides) {
  $n++
  if ($want.Count -gt 0 -and -not ($want -contains $n)) { continue }
  $sl.Export((Join-Path $OutDir ("s{0:D2}.png" -f $n)), "PNG", 1280, 720)
}
$pres.Close()
"rendered $n slides -> $OutDir"
