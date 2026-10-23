#!/usr/bin/env bash
# One-shot runner: sets up env (first run only), then detect -> interpolate -> visualize.
set -euo pipefail
cd "$(dirname "$0")"

[ -f models/license_plate_detector.pt ] || { echo "Missing models/license_plate_detector.pt (see README)"; exit 1; }
[ -f sample.mp4 ] || { echo "Missing sample.mp4 (download from https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)"; exit 1; }

if [ ! -d sort ]; then
  git clone --depth 1 https://github.com/abewley/sort.git
  sed -i '' "/matplotlib.use('TkAgg')/d" sort/sort.py  # TkAgg breaks headless/macOS runs
fi

if ! .venv/bin/python -c "import ultralytics, easyocr, filterpy" 2>/dev/null; then
  uv venv --allow-existing --python 3.10
  # unpinned on purpose: the pinned 2023 versions break on current setuptools and can't load models trained with current ultralytics
  uv pip install --python .venv ultralytics easyocr filterpy pandas scikit-image lap matplotlib
fi

PY=.venv/bin/python
$PY main.py
$PY add_missing_data.py
$PY visualize.py
echo "Done -> out.mp4"
