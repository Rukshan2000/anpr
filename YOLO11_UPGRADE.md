# Upgrade to YOLO11 and a plate OCR model on Google Colab

This guide makes plate reading more accurate with three bigger changes:

1. **A YOLO11 plate detector**, trained at a higher resolution so small and distant plates are found.
2. **A YOLO11 vehicle detector** (`yolo11s`) in place of `yolov8n`, the smallest YOLOv8 model.
3. **A plate-specific OCR model** ([fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)) in place of EasyOCR.

It also covers plate formats other than UK (Step 7).

You measure accuracy before and after every change (Steps 3 and 8), so you know what each change bought you. Training and measuring run on a free Colab GPU. Steps 9–10 bring the result back to your Mac.

Run each code block in its own Colab cell, top to bottom. You need the same accounts as in [COLAB_GUIDE.md](COLAB_GUIDE.md): Google, Roboflow, and a traffic video.

---

## Step 1: Open Colab with a GPU and mount Drive

1. Go to https://colab.research.google.com, click **New notebook**, then **Runtime → Change runtime type → T4 GPU → Save**.
2. Mount Drive, so the trained models survive the session:

```python
!nvidia-smi
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/anpr
```

## Step 2: Get the code, the models and a video

Push this project to GitHub first (`git push`), so Colab gets the current `backend/` layout.

```python
%cd /content
!git clone https://github.com/Rukshan2000/anpr.git anpr
%cd /content/anpr/backend
!pip install -q ultralytics easyocr lap roboflow "fast-plate-ocr[onnx-gpu]"
```

Upload your current plate model (`backend/models/license_plate_detector.pt` on your Mac) to `MyDrive/anpr/`. Git doesn't track `yolov8n.pt`, so download it again. Then fetch the video, trimmed to 15 s at 1080p as in COLAB_GUIDE Step 9:

```python
!cp /content/drive/MyDrive/anpr/license_plate_detector.pt models/
!wget -q -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
!mkdir -p data
!wget -q -U "Mozilla/5.0" -O data/full.mp4 "https://www.pexels.com/download/video/2103099/"
!ffmpeg -y -loglevel error -i data/full.mp4 -t 15 -vf scale=1920:-2 data/sample.mp4
```

## Step 3: Measure the baseline

You can't tell whether a change helped without a ground truth.

1. Watch `data/sample.mp4` and write down every plate you can read with your own eyes.
2. Save them in `data/truth.txt`, one per line, with no spaces.

Use your real camera footage here if you have it. Accuracy on the Pexels clip doesn't predict accuracy on your cameras.

The example below lists two plates visible in the first seconds of the Pexels clip. Replace it with every plate you can read in your video.

```python
%%writefile data/truth.txt
NA13NRU
GX15OGJ
```

Run the batch pipeline, then score it. The score cell counts how many true plates were found, and lists the wrong reads:

```python
!python -m anpr.batch data/sample.mp4
```

```python
import csv, collections
from anpr.plates import PlateVotes

def score(csv_path='data/sample.csv', min_reads=3):
    truth = set(open('data/truth.txt').read().split())
    votes = collections.defaultdict(PlateVotes)
    for row in csv.DictReader(open(csv_path)):
        votes[row['car_id']].add(row['license_number'], float(row['license_number_score']))
    read = {v.text() for v in votes.values() if v.reads >= min_reads}
    print(f'found {len(read & truth)}/{len(truth)} plates | wrong reads: {sorted(read - truth)}')

score()
```

Write the result down. Re-run these two cells after every step below.

## Step 4: Train a YOLO11 plate detector

Download the dataset and fix its paths. These are COLAB_GUIDE Steps 5–6, unchanged: add the `ROBOFLOW_API_KEY` secret first.

```python
from google.colab import userdata
from roboflow import Roboflow
import yaml

rf = Roboflow(api_key=userdata.get('ROBOFLOW_API_KEY'))
rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e").version(4).download("yolov8", location="/content/plates")

p = "/content/plates/data.yaml"
cfg = yaml.safe_load(open(p))
cfg.update(train="/content/plates/train/images", val="/content/plates/valid/images", test="/content/plates/test/images")
yaml.safe_dump(cfg, open(p, "w"))
```

Train `yolo11s` at `imgsz=960`. The current model is `yolov8n` trained at 640. A plate is only a few dozen pixels in a 1080p frame, and the extra resolution is what lets the detector find small and distant ones.

```python
!yolo detect train data=/content/plates/data.yaml model=yolo11s.pt imgsz=960 epochs=100 patience=20 batch=16 project=/content/drive/MyDrive/anpr/runs name=plates11
```

- This runs for a few hours on a T4. `patience=20` stops it early once it stops improving.
- If you get `CUDA out of memory`, use `batch=8`.
- If Colab disconnects, resume with `!yolo detect train resume model=/content/drive/MyDrive/anpr/runs/plates11/weights/last.pt`.

Compare the new detector with the old one on the same test images. Each model is scored at the image size it was trained at:

```python
!yolo detect val model=models/license_plate_detector.pt data=/content/plates/data.yaml split=test imgsz=640
!yolo detect val model=/content/drive/MyDrive/anpr/runs/plates11/weights/best.pt data=/content/plates/data.yaml split=test imgsz=960
```

Look at `mAP50-95` in the `all` row. Keep the new model only if that number is higher.

```python
!cp models/license_plate_detector.pt /content/drive/MyDrive/anpr/license_plate_detector_v8.pt   # backup
!cp /content/drive/MyDrive/anpr/runs/plates11/weights/best.pt models/license_plate_detector.pt
```

The pipeline must run the detector at the size it was trained at. In `backend/anpr/pipeline.py`, change the plate detection call in `detect()`:

```python
# before
self.plate_model(frame, verbose=False)
# after
self.plate_model(frame, imgsz=960, verbose=False)
```

In Colab you can make the change with:

```python
!sed -i 's/self.plate_model(frame, verbose=False)/self.plate_model(frame, imgsz=960, verbose=False)/' anpr/pipeline.py
```

Re-run Step 3.

## Step 5: Switch the vehicle detector to YOLO11

No training is needed. `yolo11s` is pretrained on COCO, and it uses the same class IDs (car 2, motorcycle 3, bus 5, truck 7), so `VEHICLES` stays the same.

```python
!wget -q -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
!sed -i "s/'yolov8n.pt'/'yolo11s.pt'/" anpr/pipeline.py
!grep -n "YOLO(" anpr/pipeline.py
```

Re-run Step 3. A better vehicle detector gives steadier car boxes, and the tracker splits fewer cars into two IDs. In the score cell, fewer wrong reads is the sign.

**Speed trade-off:** `yolo11s` is roughly 3× the work of `yolov8n`. On a Mac CPU the live view drops in fps. If it gets too slow, `yolo11n` is about as fast as `yolov8n` and is still more accurate.

## Step 6: Switch the OCR to fast-plate-ocr

EasyOCR is a general-purpose text reader. fast-plate-ocr is trained only on plates. On crops from the sample clip it read a blurry `M?51 VSU` plate as `MM51VSU`, where EasyOCR read `WI51VSU`, and it read the two sharp plates with about 100% confidence.

In `backend/anpr/plates.py`, replace `reader()` and `read_license_plate()` with:

```python
@functools.cache
def reader():
    from fast_plate_ocr import LicensePlateRecognizer
    return LicensePlateRecognizer('cct-s-v2-global-model')


def read_license_plate(crop):
    """Read the plate text from a BGR crop. Returns (text, score), or (None, None) if nothing plate-like was read."""
    prediction = reader().run(crop, return_confidence=True)[0]
    text = prediction.plate.upper().replace(' ', '')
    score = float(prediction.char_probs[:len(text)].mean()) if text else 0.0
    if score >= MIN_TEXT_SCORE and license_complies_format(text):
        return format_license(text), score
    return None, None
```

Notes:

- `ALLOWLIST` and `MIN_CROP_HEIGHT` were EasyOCR fixes. fast-plate-ocr resizes the crop itself and only outputs plate characters, so delete both constants.
- Its scores run much higher than EasyOCR's, so raise `MIN_TEXT_SCORE` to about `0.6`. In `pipeline.py`, raise `DONE_SCORE` to about `0.9`.
- It was trained on tight plate crops. Try `PLATE_PADDING = 0.05` in `pipeline.py` and keep whichever value scores better in Step 3.
- On your Mac, add `fast-plate-ocr[onnx]` to `backend/requirements.txt` and remove `easyocr`. Colab uses `[onnx-gpu]`.

Re-run Step 3.

## Step 7: Other plate formats (for example, Sri Lanka)

Three parts of `plates.py` assume UK plates (`AB12CDE`):

- `license_complies_format`, which rejects every other format
- `format_license`, which fixes letters and digits by their UK position
- `PlateVotes`, which votes per character over a fixed 7 positions

fast-plate-ocr's global model reads plates from many countries. To accept other formats, replace those three parts. This is the Sri Lankan version of COLAB_GUIDE Step 12e, adapted to `plates.py`:

```python
import re
from collections import Counter

# Optional province code, then 2-3 letters (new) or 2-3 digits (old), then 4 digits
SL_PLATE = re.compile(r'^(WP|CP|SP|NP|EP|NW|NC|UP|SG)?([A-Z]{2,3}|\d{2,3})(\d{4})$')


def license_complies_format(text):
    return SL_PLATE.match(format_license(text)) is not None


def format_license(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # the last 4 characters are always digits, so fix common OCR mix-ups there
    return text[:-4] + ''.join(CHAR_TO_INT.get(c, c) for c in text[-4:])


class PlateVotes:
    """Plates have different lengths here, so vote on whole plates (weighted by OCR score) instead of per character."""

    def __init__(self):
        self.votes = Counter()
        self.reads = 0

    def add(self, text, score):
        self.reads += 1
        self.votes[text] += score

    def text(self):
        return self.votes.most_common(1)[0][0]

    def score(self):
        return self.votes[self.text()] / self.reads
```

- Update the plate examples in `backend/tests/test_plates.py` to match, then run `./run.sh test`.
- Test the global OCR model on a Sri Lankan clip with Step 3.
- If it misreads often, fine-tune fast-plate-ocr on your own plate crops. This needs a CSV of crop images and their true text. Follow the training docs in the [fast-plate-ocr repo](https://github.com/ankandrew/fast-plate-ocr): `pip install "fast-plate-ocr[train]"`, then `fast-plate-ocr train --help`.
- To also retrain the detector on Sri Lankan plates, use COLAB_GUIDE Steps 12a–12d with `model=/content/drive/MyDrive/anpr/runs/plates11/weights/best.pt imgsz=960`.
- Sinhala characters (`ශ්‍රී`) on older plates still can't be read.

## Step 8: Final comparison

Run Step 3 one last time and compare it with the baseline. Then check the annotated video by eye:

```python
from google.colab import files
files.download('data/sample_out.mp4')
```

If a step made the score worse on your footage, undo that step. Not every change helps every camera.

## Step 9: Save everything to Drive

```python
!cp models/license_plate_detector.pt models/yolo11s.pt /content/drive/MyDrive/anpr/
!cd /content/anpr && git diff > /content/drive/MyDrive/anpr/yolo11_changes.patch
```

## Step 10: Bring it back to your Mac

1. From Drive, copy `license_plate_detector.pt` and `yolo11s.pt` into `backend/models/`.
2. Apply the code changes. Either edit the same lines by hand, or apply the patch from the repo root: `git apply ~/Downloads/yolo11_changes.patch`.
3. Install the new dependency: `.venv/bin/pip install "fast-plate-ocr[onnx]"`, or `uv pip install --python .venv -r backend/requirements.txt` after editing that file.
4. Run `./run.sh test`, then `./run.sh dev`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: anpr` | You're in the wrong folder. Run `%cd /content/anpr/backend`. |
| `CUDA out of memory` while training | Use `batch=8`. |
| Plates found, but every read is rejected | `MIN_TEXT_SCORE` is too high, or the plate format check doesn't match your country's plates (Step 7). |
| New detector finds fewer plates than the old one | The pipeline is running it at the wrong size. Check that `imgsz` in `detect()` matches the training `imgsz`. |
| Live view on the Mac is too slow | Use `yolo11n.pt` for vehicles, or plate `imgsz=640` with a detector retrained at 640. |
| `onnxruntime` error on the Mac | Install `fast-plate-ocr[onnx]`, not `[onnx-gpu]`. |
