# Run ANPR on Google Colab: Step 0 to Hero

This guide trains your own license plate detector on a free Colab GPU and then runs the full pipeline: detect, track, read plates, interpolate, and render `out.mp4`.

Run each code block in its own Colab cell, top to bottom.

---

## Step 0: What you need

- A Google account, for Colab and Drive.
- A free [Roboflow](https://roboflow.com) account, used to download the dataset.
- A traffic video. The tutorial uses [this Pexels clip](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/). Download it to your computer first.

## Step 1: Open Colab with a GPU

1. Go to https://colab.research.google.com and click **New notebook**.
2. Open **Runtime → Change runtime type**, choose **T4 GPU**, and click **Save**.
3. Check that the GPU is attached:

```python
!nvidia-smi
```

You should see a Tesla T4. If you get "command not found", go back and repeat step 2.

## Step 2: Mount Google Drive

Colab deletes everything when the session ends. Save the model and outputs to Drive so you keep them.

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/anpr
```

## Step 3: Get the code and the SORT tracker

```python
%cd /content
!git clone https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8.git anpr
%cd /content/anpr
!git clone --depth 1 https://github.com/abewley/sort.git
# SORT forces a desktop GUI backend, which crashes on Colab. Remove that line.
!sed -i "/matplotlib.use('TkAgg')/d" sort/sort.py
```

## Step 4: Install dependencies

Colab runs a newer Python than the old versions pinned in `requirements.txt`, so install the packages without pins:

```python
!pip install -q ultralytics easyocr filterpy scikit-image lap roboflow
```

If Colab asks you to **Restart session**, restart it, then run `%cd /content/anpr` before you continue.

## Step 5: Add your Roboflow API key as a secret

1. On Roboflow, open **Settings → API Keys** (or https://app.roboflow.com/settings/api) and copy your private key.
2. In Colab, click the **🔑 key icon** in the left sidebar, then **Add new secret**.
   - Name: `ROBOFLOW_API_KEY`
   - Value: your key
   - Turn on **Notebook access**.

This keeps the key out of the notebook, so it isn't shared if you share the notebook.

## Step 6: Download the license plate dataset

```python
from google.colab import userdata
from roboflow import Roboflow

rf = Roboflow(api_key=userdata.get('ROBOFLOW_API_KEY'))
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(4).download("yolov8", location="/content/plates")
```

Roboflow's `data.yaml` uses relative paths that often confuse YOLO. Replace them with absolute paths:

```python
import yaml
p = "/content/plates/data.yaml"
cfg = yaml.safe_load(open(p))
cfg["train"] = "/content/plates/train/images"
cfg["val"]   = "/content/plates/valid/images"
cfg["test"]  = "/content/plates/test/images"
yaml.safe_dump(cfg, open(p, "w"))
print(cfg)
```

## Step 7: Train the plate detector

```python
!yolo detect train data=/content/plates/data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16 project=/content/drive/MyDrive/anpr/runs name=plates
```

- On a T4 this takes roughly 1–2 hours. Keep the browser tab open, because Colab disconnects idle sessions.
- If you only want a quick test, use `epochs=10`.
- Training saves to Drive, so a disconnect doesn't lose finished epochs. To continue from where it stopped:

```python
!yolo detect train resume model=/content/drive/MyDrive/anpr/runs/plates/weights/last.pt
```

When training finishes, check `mAP50` in the last lines of the log. A value above 0.9 is good.

## Step 8: Put the model where `main.py` expects it

```python
!mkdir -p /content/anpr/models
!cp /content/drive/MyDrive/anpr/runs/plates/weights/best.pt /content/anpr/models/license_plate_detector.pt
```

## Step 9: Get the video

The easiest way is to download the tutorial's [Pexels traffic video](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/) straight into Colab (4K, about 184 MB):

```python
!wget -q --show-progress -U "Mozilla/5.0" -O /content/anpr/sample.mp4 "https://www.pexels.com/download/video/2103099/"
```

The video is 4K, and the scripts process every frame, so `main.py` would be very slow on the full clip. Right after the download, cut it to 15 seconds and scale it down to 1080p:

```python
!ffmpeg -y -i /content/anpr/sample.mp4 -t 15 -vf scale=1920:-2 /content/anpr/clip.mp4 && mv /content/anpr/clip.mp4 /content/anpr/sample.mp4
```

### Other ways to get a video

**Upload from your computer.** If you'd rather use the browser, open the [Pexels page](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/), click **Free download**, then upload the file (any traffic video works):

```python
from google.colab import files
up = files.upload()  # pick your traffic video
import shutil
shutil.move(next(iter(up)), "/content/anpr/sample.mp4")
```

**Copy from Google Drive** (the video is already in Drive):

```python
!cp "/content/drive/MyDrive/anpr/your_video.mp4" /content/anpr/sample.mp4
```

Whichever way you choose, the file must be named exactly `sample.mp4`, because the scripts hardcode it. Run the trim cell above on any long or 4K video.

## Step 10: Run the pipeline

```python
%cd /content/anpr
!python main.py              # detect + track + OCR → test.csv
!python add_missing_data.py  # fill gaps between frames → test_interpolated.csv
!python visualize.py         # draw boxes and plates → out.mp4
```

`main.py` prints one line per frame, so long output is normal.

## Step 11: Get your results

```python
!cp /content/anpr/out.mp4 /content/anpr/test_interpolated.csv /content/drive/MyDrive/anpr/
from google.colab import files
files.download("/content/anpr/out.mp4")
```

Your trained model is saved in `MyDrive/anpr/runs/plates/weights/best.pt`. Copy it into the local project as `backend/models/license_plate_detector.pt` and run `./run.sh dev` on your Mac (see the README). For Sri Lankan plates locally, apply the Step 12e changes to `backend/anpr/plates.py`.

---

## Step 12: Make it work for Sri Lankan plates

Two things need to change for Sri Lankan vehicles:

1. **The detector** has only seen the Roboflow dataset. Fine-tuning it on the [Sri Lanka vehicle number plates](https://www.kaggle.com/datasets/rohankumara/sri-lanka-vehicle-number-plates) dataset (Kaggle, CC0) teaches it what local plates look like.
2. **The OCR filter.** The original `util.py` only accepts UK-style plates like `AB12CDE` and throws everything else away, so it would never output a Sri Lankan plate. Step 12e fixes this. After that change, UK plates are no longer read.

### 12a. Add your Kaggle API key as secrets

1. On Kaggle, go to **Settings → API → Create New Token**. This downloads `kaggle.json`; open it to see your `username` and `key`.
2. In Colab, open the **🔑** sidebar and add two secrets, each with **Notebook access** turned on:
   - `KAGGLE_USERNAME`: your username
   - `KAGGLE_KEY`: your key

### 12b. Download the dataset

```python
import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
!pip install -q kaggle
!kaggle datasets download -d rohankumara/sri-lanka-vehicle-number-plates -p /content/sl --unzip
!ls /content/sl/Numberplate | head
```

You should see pairs like `100.jpg` and `100.xml`.

### 12c. Convert the labels to YOLO format

The dataset uses Pascal VOC `.xml` labels, but YOLO needs one `.txt` per image. This cell converts them, puts 20% of the images aside for validation, and writes `data.yaml`:

```python
import glob, os, random, shutil, cv2
import xml.etree.ElementTree as ET

src, dst = "/content/sl", "/content/sl_yolo"
xmls = sorted(glob.glob(f"{src}/**/*.xml", recursive=True))
random.seed(0)
random.shuffle(xmls)
n_val = len(xmls) // 5

kept = 0
for i, xml_path in enumerate(xmls):
    split = "val" if i < n_val else "train"
    img_path = xml_path[:-4] + ".jpg"
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    lines = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # every box is a plate, so everything is class 0
        lines.append(f"0 {(x1 + x2) / 2 / w} {(y1 + y2) / 2 / h} {(x2 - x1) / w} {(y2 - y1) / h}")

    name = os.path.basename(img_path)[:-4]
    os.makedirs(f"{dst}/images/{split}", exist_ok=True)
    os.makedirs(f"{dst}/labels/{split}", exist_ok=True)
    shutil.copy(img_path, f"{dst}/images/{split}/{name}.jpg")
    open(f"{dst}/labels/{split}/{name}.txt", "w").write("\n".join(lines))
    kept += 1

open(f"{dst}/data.yaml", "w").write(f"path: {dst}\ntrain: images/train\nval: images/val\nnames:\n  0: License_Plate\n")
print(kept, "images converted,", n_val, "for validation")
```

### 12d. Fine-tune your plate detector

Start from the model you trained in Step 7 instead of from scratch. It already knows plates in general, so it only has to learn the Sri Lankan look:

```python
!yolo detect train data=/content/sl_yolo/data.yaml model=/content/drive/MyDrive/anpr/runs/plates/weights/best.pt epochs=50 imgsz=640 batch=16 project=/content/drive/MyDrive/anpr/runs name=sl_plates
!cp /content/drive/MyDrive/anpr/runs/sl_plates/weights/best.pt /content/anpr/models/license_plate_detector.pt
```

This dataset is small (about 20 MB), so training takes minutes, not hours.

### 12e. Switch the OCR to Sri Lankan plates and the GPU

This cell adds the Sri Lankan reader to the end of it (the new functions replace the UK ones). It also moves EasyOCR onto the GPU, which makes `main.py` much faster:

```python
%cd /content/anpr
!sed -i 's/gpu=False/gpu=True/' util.py
```

```python
%%writefile -a /content/anpr/util.py


import re

# Optional province code, then 2-3 letters (new) or 2-3 digits (old), then 4 digits
SL_PLATE = re.compile(r'^(WP|CP|SP|NP|EP|NW|NC|UP|SG)?([A-Z]{2,3}|\d{2,3})(\d{4})$')


def parse_sri_lankan_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # the last 4 characters are always digits, so fix common OCR mix-ups there
    text = text[:-4] + ''.join(dict_char_to_int.get(c, c) for c in text[-4:])
    match = SL_PLATE.match(text)
    if match is None:
        return None
    province, series, number = match.groups()
    return ' '.join(filter(None, [province, '{}-{}'.format(series, number)]))


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None, None
    # two-line plates (bikes, three-wheelers) come back as one detection per line
    text = parse_sri_lankan_plate(''.join(detection[1] for detection in detections))
    if text is None:
        return None, None
    return text, min(detection[2] for detection in detections)
```

Run this cell only once. If you run it again, it adds a second copy to the end of the file. That copy is harmless, but if you want a clean file, re-clone the repo (Step 3).

### 12f. Run it on a Sri Lankan video

Download or upload a Sri Lankan traffic video as `sample.mp4` (Step 9), then run Step 10 again. Plates are written like `WP CAB-1234`, `KX-1234` or `250-1234`.

**What it can't read:** older plates with the Sinhala `ශ්‍රී` character. EasyOCR runs with English only here, so those plates are skipped.

---

## Next time (skip training)

After you have trained the model once, a new session only needs Steps 1–4, then:

```python
!mkdir -p /content/anpr/models
!cp /content/drive/MyDrive/anpr/runs/plates/weights/best.pt /content/anpr/models/license_plate_detector.pt
```

If you did Step 12, copy `runs/sl_plates/weights/best.pt` instead, and run Step 12e again, because a fresh clone has the original `util.py`.

Then do Steps 9–11.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: sort` | You're in the wrong folder. Run `%cd /content/anpr`. |
| `TkAgg` / `tkinter` error | Run the `sed` line from Step 3 again. |
| `FileNotFoundError ... data.yaml` paths | Run the yaml fix in Step 6 again. |
| `CUDA out of memory` while training | Use `batch=8`. |
| `out.mp4` has no plate text | The plates may be too small or blurry in that video, or the model needs more epochs. Try the Pexels clip first. |
| Session disconnected | Remount Drive, redo Steps 3–4, and resume training (Step 7). |
| `test.csv` is empty | No plate was read on any tracked car. Check that the model is the trained `best.pt` and not `yolov8n.pt`. |
