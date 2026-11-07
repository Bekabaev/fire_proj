#!/usr/bin/env bash
# Установка без sudo (проверено на Debian 12, 07.11.2026): gfortran, OpenMPI, GDAL и python-пакеты - в conda-окружение
# "elmfire", ELMFIRE - клон в ~/elmfire, сборка только основного бинарника. Запуск из корня репозитория:
#   bash elmfire_bench/setup_conda.sh
# Потом в каждой новой оболочке:
#   source ~/miniconda3/bin/activate elmfire
#   export PATH=$PATH:~/elmfire/build/linux/bin OMPI_MCA_osc=^ucx OMPI_MCA_pml=^ucx     (^ucx - убрать шум UCX в логах)
#   python elmfire_bench/run_elmfire.py && python elmfire_bench/evaluate.py
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ELM="${ELMFIRE_BASE_DIR:-$HOME/elmfire}"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"

echo "== 1. conda-окружение elmfire"
"$CONDA" env list | grep -q "^elmfire " || "$CONDA" create -y -q -n elmfire -c conda-forge --override-channels \
    python=3.11 gfortran openmpi gdal numpy scipy rasterio matplotlib python-pptx
source "$(dirname "$CONDA")/activate" elmfire

echo "== 2. ELMFIRE -> $ELM"
[ -d "$ELM/.git" ] || git clone https://github.com/lautenberger/elmfire.git "$ELM"
git -C "$ELM" log -1 --format='ELMFIRE commit %h от %cs'

echo "== 3. сборка (только elmfire_<версия>)"
export ELMFIRE_BASE_DIR="$ELM" ELMFIRE_INSTALL_DIR="$ELM/build/linux/bin"
"$ELM/build/linux/make_gnu.sh" fast
ls -la "$ELMFIRE_INSTALL_DIR"

echo "== 4. проверка оценки"
cd "$REPO" && python elmfire_bench/evaluate.py | tail -12
