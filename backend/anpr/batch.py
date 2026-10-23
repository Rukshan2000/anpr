"""
Offline pipeline for a video file: detect -> interpolate -> render.

usage (from backend/): python -m anpr.batch data/sample.mp4
writes data/sample.csv, data/sample_interpolated.csv and data/sample_out.mp4
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from .plates import PlateVotes

FIELDS = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
          'license_number_score']


def fmt_bbox(bbox):
    return '[{}]'.format(' '.join(str(float(v)) for v in bbox))


def parse_bbox(text):
    return [float(v) for v in text.strip('[]').split()]


def detect(video_path, csv_path):
    """Run detection + tracking + OCR on every frame; write one row per car with a readable plate per frame."""
    from .pipeline import Pipeline  # loads the models, keep it out of interpolate/render

    pipeline = Pipeline()
    cap = cv2.VideoCapture(str(video_path))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        frame_nmr = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_nmr += 1
            _, readings = pipeline.detect(frame)
            for r in {r.car_id: r for r in readings if r.text is not None}.values():  # last reading per car wins
                writer.writerow({'frame_nmr': frame_nmr, 'car_id': r.car_id, 'car_bbox': fmt_bbox(r.car_bbox),
                                 'license_plate_bbox': fmt_bbox(r.plate_bbox), 'license_plate_bbox_score': r.plate_score,
                                 'license_number': r.text, 'license_number_score': r.text_score})
            print('frame', frame_nmr, end='\r')
    cap.release()


def interpolate(rows):
    """Fill the frames where a tracked car had no plate reading by linearly interpolating its boxes."""
    by_car = defaultdict(list)
    for row in rows:
        by_car[row['car_id']].append(row)

    out = []
    for car_id, car_rows in by_car.items():
        car_rows.sort(key=lambda r: int(r['frame_nmr']))
        for prev, row in zip([None] + car_rows, car_rows):
            if prev is not None:
                a, b = int(prev['frame_nmr']), int(row['frame_nmr'])
                for frame_nmr in range(a + 1, b):
                    t = (frame_nmr - a) / (b - a)
                    lerp = lambda key: fmt_bbox(p + (q - p) * t for p, q in zip(parse_bbox(prev[key]), parse_bbox(row[key])))
                    out.append({'frame_nmr': frame_nmr, 'car_id': car_id, 'car_bbox': lerp('car_bbox'),
                                'license_plate_bbox': lerp('license_plate_bbox'), 'license_plate_bbox_score': 0,
                                'license_number': '0', 'license_number_score': 0})
            out.append(row)
    return out


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    for (cx, cy), dx, dy in [((x1, y1), 1, 1), ((x1, y2), 1, -1), ((x2, y1), -1, 1), ((x2, y2), -1, -1)]:
        cv2.line(img, (cx, cy), (cx, cy + dy * line_length_y), color, thickness)
        cv2.line(img, (cx, cy), (cx + dx * line_length_x, cy), color, thickness)
    return img


def render(video_path, rows, out_path):
    """Draw car corners, plate boxes and each car's best plate (crop + text) above it."""
    cap = cv2.VideoCapture(str(video_path))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # per car: the plate text voted over all its reads, and a crop of the plate from its best-scoring frame
    best, votes = {}, defaultdict(PlateVotes)
    for row in rows:
        if float(row['license_number_score']) > 0:  # skip interpolated rows
            votes[row['car_id']].add(row['license_number'], float(row['license_number_score']))
        if row['car_id'] not in best or float(row['license_number_score']) > float(best[row['car_id']]['license_number_score']):
            best[row['car_id']] = row
    plates = {}
    for car_id, row in best.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_nmr']))
        _, frame = cap.read()
        x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        plates[car_id] = (votes[car_id].text(), cv2.resize(crop, (int((x2 - x1) * 400 / (y2 - y1)), 400)))

    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row['frame_nmr'])].append(row)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_nmr = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1
        for row in by_frame[frame_nmr]:
            car_x1, car_y1, car_x2, car_y2 = map(int, parse_bbox(row['car_bbox']))
            draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25)
            x1, y1, x2, y2 = map(int, parse_bbox(row['license_plate_bbox']))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

            text, crop = plates[row['car_id']]
            H, W, _ = crop.shape
            left, right = (car_x2 + car_x1 - W) // 2, (car_x2 + car_x1 - W) // 2 + W
            try:
                frame[car_y1 - H - 100:car_y1 - 100, left:right] = crop
                frame[car_y1 - H - 400:car_y1 - H - 100, left:right] = (255, 255, 255)
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                cv2.putText(frame, text, ((car_x2 + car_x1 - text_width) // 2, car_y1 - H - 250 + text_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
            except ValueError:
                pass  # overlay doesn't fit inside the frame (car near the top edge)
        out.write(frame)

    out.release()
    cap.release()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    video = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/sample.mp4')
    raw, interpolated, out = (video.with_name(video.stem + suffix) for suffix in ('.csv', '_interpolated.csv', '_out.mp4'))
    detect(video, raw)
    rows = interpolate(read_csv(raw))
    write_csv(interpolated, rows)
    render(video, rows, out)
    print('Done ->', out)
