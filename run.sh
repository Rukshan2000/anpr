#!/usr/bin/env bash
# usage: ./run.sh dev            API on :8000 + React dev server on :5173 (open http://localhost:5173)
#        ./run.sh serve          build the React app and serve everything from the API on :8000
#        ./run.sh batch [video]  offline detect -> interpolate -> render (path relative to backend/, default data/sample.mp4)
#        ./run.sh test
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
PY=$ROOT/.venv/bin/python

[ -f backend/models/license_plate_detector.pt ] || { echo "Missing backend/models/license_plate_detector.pt (see README)"; exit 1; }

# first run only: Python env, node modules
if ! "$PY" -c "import ultralytics, easyocr, lap, fastapi, uvicorn" 2>/dev/null; then
  uv venv --allow-existing --python 3.10
  uv pip install --python .venv -r backend/requirements.txt
fi
[ -d frontend/node_modules ] || npm --prefix frontend install

cd backend
case "${1:-dev}" in
  dev)
    trap 'kill 0' EXIT
    "$PY" -m uvicorn api.main:app --reload --reload-dir api --reload-dir anpr &
    npm --prefix "$ROOT/frontend" run dev
    ;;
  serve)
    npm --prefix "$ROOT/frontend" run build
    echo "Open http://127.0.0.1:8000"
    "$PY" -m uvicorn api.main:app
    ;;
  batch)
    [ -f "${2:-data/sample.mp4}" ] || { echo "Missing ${2:-data/sample.mp4} (download from https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)"; exit 1; }
    "$PY" -m anpr.batch "${2:-data/sample.mp4}"
    ;;
  test)
    PYTHONPATH=. "$PY" tests/test_plates.py
    ;;
  *) sed -n '2,5p' "$ROOT/run.sh"; exit 1 ;;
esac
