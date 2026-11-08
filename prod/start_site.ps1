# Запуск сайта, мониторинга FIRMS, Telegram-бота и публичного туннеля (например, после отключения света).
#   powershell -ExecutionPolicy Bypass -File prod\start_site.ps1
# Сервер: http://127.0.0.1:8000 (лог prod\server_log.txt, мониторинг prod\monitor_log.txt).
# Туннель Cloudflare: адрес *.trycloudflare.com меняется при каждом запуске - скрипт печатает новый;
# бот берёт его из tools\tunnel_log.txt сам, руководителю новую ссылку нужно переслать.
$ErrorActionPreference = "Stop"
$proj  = Split-Path -Parent $PSScriptRoot                    # project_till_nov
$root  = Split-Path -Parent $proj                            # burned_area
$py    = Join-Path $root ".venv\Scripts\python.exe"
$cf    = Join-Path $root "tools\cloudflared.exe"
$tlog  = Join-Path $root "tools\tunnel_log.txt"

# сервер (если порт 8000 уже занят - считаем, что он запущен)
$busy = $null
try { $busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction Stop } catch {}
if (-not $busy) {
    Start-Process -FilePath $py -WorkingDirectory $proj -WindowStyle Hidden `
        -ArgumentList "-X utf8 -W ignore -m uvicorn prod.server:app --host 127.0.0.1 --port 8000" `
        -RedirectStandardOutput (Join-Path $proj "prod\server_out.txt") -RedirectStandardError (Join-Path $proj "prod\server_log.txt")
    Write-Host "сервер запускается..."
} else { Write-Host "сервер уже работает" }

# туннель
if (-not (Get-Process cloudflared -ErrorAction SilentlyContinue)) {
    if (Test-Path $tlog) { Remove-Item $tlog -Force }
    Start-Process -FilePath $cf -WindowStyle Hidden -ArgumentList "tunnel --no-autoupdate --url http://127.0.0.1:8000" -RedirectStandardError $tlog
    $url = $null
    for ($i = 0; $i -lt 60 -and -not $url; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Path $tlog) { $m = Select-String -Path $tlog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -Last 1; if ($m) { $url = $m.Matches[0].Value } }
    }
    if ($url) { Write-Host "публичная ссылка: $url" } else { Write-Host "туннель не ответил, см. $tlog" }
} else {
    $m = Select-String -Path $tlog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -Last 1
    Write-Host "туннель уже работает: $($m.Matches[0].Value)"
}
