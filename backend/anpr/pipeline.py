import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from ultralytics import YOLO

from .plates import PlateVotes, get_car, read_license_plate

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
VEHICLES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
PLATE_PADDING = 0.15  # grow plate boxes by this fraction of their height: the detector often clips edge characters
# stop running OCR on a car once its plate is this settled; saves time for the other cars and frames
DONE_READS, DONE_SCORE = 20, 0.6


class Reading(NamedTuple):
    car_id: int
    car_bbox: list
    plate_bbox: list
    plate_score: float
    text: str | None
    text_score: float | None


class Pipeline:
    """Vehicle detection + tracking + plate reading. Keeps tracker state, so use one instance per video."""

    def __init__(self):
        self.coco_model = YOLO(str(MODELS_DIR / 'yolov8n.pt'))
        self.plate_model = YOLO(str(MODELS_DIR / 'license_plate_detector.pt'))
        self.reset()

    def reset(self):
        """Forget tracked cars and plate readings, e.g. when switching to another video."""
        self.new_video = True  # makes the next track() call start a fresh tracker
        self.votes = {}  # car_id -> PlateVotes
        self.best_plate = {}  # car_id -> (text, score, reads, time): snapshot of votes, safe to read from other threads

    def track(self, frame):
        """Return tracked vehicles as an array of rows (x1, y1, x2, y2, car_id)."""
        boxes = self.coco_model.track(frame, persist=not self.new_video, classes=VEHICLES, tracker='bytetrack.yaml',
                                      verbose=False)[0].boxes
        self.new_video = False
        if boxes.id is None:
            return np.empty((0, 5))
        return np.column_stack([boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()])

    def detect(self, frame, skip_ocr=()):
        """Return (tracks, readings) for one frame: tracked cars and the plates found on them."""
        tracks = self.track(frame)
        height, width = frame.shape[:2]

        readings = []
        for x1, y1, x2, y2, score, _ in self.plate_model(frame, verbose=False)[0].boxes.data.tolist():
            car = get_car((x1, y1, x2, y2), tracks)
            if car is None:
                continue
            car_id = int(car[4])
            text, text_score = None, None
            if car_id not in skip_ocr:
                pad = (y2 - y1) * PLATE_PADDING
                crop = frame[max(int(y1 - pad), 0):min(int(y2 + pad), height), max(int(x1 - pad), 0):min(int(x2 + pad), width)]
                if crop.size:
                    text, text_score = read_license_plate(crop)
            readings.append(Reading(car_id, car[:4].tolist(), [x1, y1, x2, y2], score, text, text_score))
        return tracks, readings

    def process(self, frame):
        """Detect, vote on plates and draw the results onto the frame."""
        done = {car_id for car_id, (_, score, reads, _) in self.best_plate.items() if reads >= DONE_READS and score >= DONE_SCORE}
        tracks, readings = self.detect(frame, skip_ocr=done)

        for r in readings:
            x1, y1, x2, y2 = map(int, r.plate_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if r.text is None:
                continue
            votes = self.votes.setdefault(r.car_id, PlateVotes())
            old_text = votes.text() if votes.reads else None
            votes.add(r.text, r.text_score)
            if votes.text() != old_text:
                print('car {}: {} ({:.2f}, {} reads)'.format(r.car_id, votes.text(), votes.score(), votes.reads))
            self.best_plate[r.car_id] = (votes.text(), votes.score(), votes.reads, time.time())

        for xcar1, ycar1, xcar2, ycar2, car_id in tracks:
            cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 3)
            if int(car_id) in self.best_plate:
                cv2.putText(frame, self.best_plate[int(car_id)][0], (int(xcar1), int(ycar1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        return frame
