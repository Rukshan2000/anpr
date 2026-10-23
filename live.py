import os
import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from sort.sort import Sort
from util import get_car, read_license_plate

# CCTV cameras drop packets over UDP, which smears frames
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
mot_tracker = Sort()

vehicles = [2, 3, 5, 7]

best_plate = {}  # car_id -> (text, score, time): keep the most confident reading per tracked car


def parse_source(text):
    return int(text) if text.isdigit() else text


def open_capture(source):
    if isinstance(source, str):
        # give up on a dead camera after 5 s instead of hanging for ~30 s
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
    return cv2.VideoCapture(source)


def frames(source):
    """
    Yield frames from a webcam (int), a stream URL or a video file.

    A video file is read frame by frame. A live stream keeps producing frames while we process one, so for
    streams always yield the newest frame instead of falling further and further behind, and reconnect
    when the stream drops (network, camera reboot).
    """
    if os.path.isfile(str(source)):
        cap = cv2.VideoCapture(source)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    return
                yield frame
        finally:
            cap.release()

    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError('Could not open video source: {}'.format(source))

    latest = {'frame': None, 'stop': False}

    def grab_frames(cap):
        while not latest['stop']:
            ret, frame = cap.read()
            if ret:
                latest['frame'] = frame
                continue
            print('Stream lost, reconnecting...')
            cap.release()
            time.sleep(2)
            cap = open_capture(source)
        cap.release()

    threading.Thread(target=grab_frames, args=(cap,), daemon=True).start()
    try:
        while True:
            if latest['frame'] is None:
                time.sleep(0.01)
                continue
            frame, latest['frame'] = latest['frame'], None
            yield frame
    finally:
        latest['stop'] = True


def reset():
    """Forget tracked cars and plate readings, e.g. when switching to another camera."""
    global mot_tracker
    mot_tracker = Sort()
    best_plate.clear()


def process(frame):
    """Detect, track and read plates in one frame. Draws the results onto the frame and updates best_plate."""

    # detect and track vehicles
    detections_ = [d[:5] for d in coco_model(frame, verbose=False)[0].boxes.data.tolist() if int(d[5]) in vehicles]
    track_ids = mot_tracker.update(np.asarray(detections_) if detections_ else np.empty((0, 5)))

    # detect license plates
    for license_plate in license_plate_detector(frame, verbose=False)[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        if car_id == -1:
            continue

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # crop and process license plate, same as main.py
        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
        if license_plate_crop.size == 0:
            continue
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
        car_id = int(car_id)
        if license_plate_text is not None and license_plate_text_score > best_plate.get(car_id, ('', 0))[1]:
            if best_plate.get(car_id, ('',))[0] != license_plate_text:
                print('car {}: {} ({:.2f})'.format(car_id, license_plate_text, license_plate_text_score))
            best_plate[car_id] = (license_plate_text, license_plate_text_score, time.time())

    # draw tracked cars with their best plate reading
    for xcar1, ycar1, xcar2, ycar2, car_id in track_ids:
        cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 3)
        if int(car_id) in best_plate:
            cv2.putText(frame, best_plate[int(car_id)][0], (int(xcar1), int(ycar1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return frame


if __name__ == '__main__':
    # usage: python live.py rtsp://user:pass@camera-ip:554/stream   (or 0 for a webcam, or a video file)
    # or set CCTV_URL in the environment and run python live.py, to keep the camera password out of the command
    source = parse_source(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('CCTV_URL', '0'))
    try:
        for frame in frames(source):
            cv2.imshow('ANPR live (press q to quit)', process(frame))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except RuntimeError as e:
        sys.exit(str(e))
    cv2.destroyAllWindows()
