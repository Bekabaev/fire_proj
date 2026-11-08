# Обновить содержание и поля docx через Word (Windows). Использование: powershell -File update_toc.ps1 <file.docx> [out.pdf]
param([string]$File, [string]$Pdf = "")
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$doc = $w.Documents.Open((Resolve-Path $File).Path)
foreach ($t in $doc.TablesOfContents) { $t.Update() }
$doc.Fields.Update() | Out-Null
foreach ($t in $doc.TablesOfContents) { $t.Update() }
$doc.Save()
if ($Pdf) { $doc.ExportAsFixedFormat($Pdf, 17) }
"pages: " + $doc.ComputeStatistics(2)
$doc.Close(); $w.Quit()
