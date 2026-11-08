#!/usr/bin/env bash
# Установка всего для сравнения с ELMFIRE на Debian/Ubuntu (ноутбук). Запуск из корня репозитория:
#   bash elmfire_bench/setup_debian.sh
# Делает: системные пакеты (gfortran, OpenMPI, GDAL, python3) -> клон ELMFIRE в ~/elmfire -> сборка -> python-venv
# с numpy/scipy/rasterio в elmfire_bench/.venv -> проверка оценки на наших моделях. Повторный запуск безопасен.
# ВНИМАНИЕ: написан на Windows-ПК и на Debian не проверялся; если шаг сборки упадёт - смотреть ~/elmfire/README.md
# и build/linux/ в репозитории ELMFIRE (там скрипты сборки), поправить шаг 3 и закоммитить исправление.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ELM="${ELMFIRE_BASE_DIR:-$HOME/elmfire}"

echo "== 1. системные пакеты (нужен sudo)"
sudo apt-get update
sudo apt-get install -y git build-essential gfortran openmpi-bin libopenmpi-dev gdal-bin libgdal-dev \
    python3 python3-venv python3-pip python3-gdal bc

echo "== 2. ELMFIRE -> $ELM"
if [ ! -d "$ELM/.git" ]; then
    git clone https://github.com/lautenberger/elmfire.git "$ELM"
else
    git -C "$ELM" pull --ff-only || true
fi
git -C "$ELM" log -1 --format='ELMFIRE commit %h от %cs'

echo "== 3. сборка ELMFIRE"
export ELMFIRE_BASE_DIR="$ELM"
export ELMFIRE_INSTALL_DIR="$ELM/build/linux/bin"
export ELMFIRE_SCRATCH_BASE="${ELMFIRE_SCRATCH_BASE:-/tmp/elmfire_scratch}"
mkdir -p "$ELMFIRE_SCRATCH_BASE"
cd "$ELM/build/linux"
if [ -x ./make_gnu.sh ]; then
    ./make_gnu.sh
else
    echo "!! нет build/linux/make_gnu.sh - собрать по README ELMFIRE и поправить этот шаг"; ls; exit 1
fi
ls -la "$ELMFIRE_INSTALL_DIR" || true
cd "$REPO"

echo "== 4. переменные окружения -> ~/.bashrc"
grep -q ELMFIRE_BASE_DIR ~/.bashrc || cat >> ~/.bashrc <<EOF
export ELMFIRE_BASE_DIR="$ELM"
export ELMFIRE_INSTALL_DIR="$ELM/build/linux/bin"
export ELMFIRE_SCRATCH_BASE="$ELMFIRE_SCRATCH_BASE"
export PATH="\$PATH:$ELM/build/linux/bin:$ELM/cloudfire"
EOF

echo "== 5. python-окружение для оценки"
python3 -m venv "$REPO/elmfire_bench/.venv"
"$REPO/elmfire_bench/.venv/bin/pip" install -q --upgrade pip numpy scipy rasterio
"$REPO/elmfire_bench/.venv/bin/python" "$REPO/elmfire_bench/evaluate.py" | tail -9

echo "== готово. Дальше: tutorials в $ELM/tutorials (начать с первого), затем elmfire_bench/run_elmfire.py (см. HANDOFF.md)"
