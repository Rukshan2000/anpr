# automatic-number-plate-recognition-python-yolov8

<p align="center">
<a href="https://www.youtube.com/watch?v=fyJB1t0o0ms">
    <img width="600" src="https://utils-computervisiondeveloper.s3.amazonaws.com/thumbnails/with_play_button/anpr_yolo2.jpg" alt="Watch the video">
    </br>Watch on YouTube: Automatic number plate recognition with Python, Yolov8 and EasyOCR !
</a>
</p>

Vehicles are detected with YOLOv8, tracked with ByteTrack, plates detected with a custom YOLOv8 model and read with EasyOCR.
A FastAPI backend runs the pipeline on a webcam, CCTV stream or video file; a React app shows the live video and plates.

## Layout

```
backend/
  anpr/             detection pipeline (no web code)
    plates.py       OCR + plate format rules
    pipeline.py     detect, track, read plates on a frame, vote per car
    sources.py      webcam / file / RTSP frame reader with reconnect
    batch.py        offline: video -> csv -> interpolated csv -> annotated video
  api/
    main.py         FastAPI: /api/status, /api/source, /api/stream (MJPEG)
    camera.py       background worker running the pipeline on the current source
  models/           license_plate_detector.pt (+ yolov8n.pt, downloaded on first run)
  data/             your videos and batch outputs (git-ignored)
  tests/
frontend/           Vite + React UI
run.sh
```

## Run

Needs `uv`, Node 20+ and `backend/models/license_plate_detector.pt`. The first run installs everything.

```bash
./run.sh dev                    # API :8000 + UI with hot reload → http://localhost:5173
./run.sh serve                  # build UI, serve everything → http://127.0.0.1:8000
./run.sh batch data/sample.mp4  # writes data/sample.csv, data/sample_interpolated.csv, data/sample_out.mp4
./run.sh test
```

Set `CCTV_URL=rtsp://user:pass@camera-ip:554/stream` to pick the starting source; it can be changed from the page.
The API only listens on 127.0.0.1, because a source can be any local file or URL.

## data

The video used in the tutorial can be downloaded [here](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/). Save it as `backend/data/sample.mp4`.

## models

A Yolov8 pretrained model was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) and following this [step by step tutorial on how to train an object detector with Yolov8 on your custom data](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide). See [COLAB_GUIDE.md](COLAB_GUIDE.md) to train your own, and [YOLO11_UPGRADE.md](YOLO11_UPGRADE.md) to upgrade to YOLO11 and a plate OCR model for better accuracy.

The trained model is available in my [Patreon](https://www.patreon.com/ComputerVisionEngineer).
