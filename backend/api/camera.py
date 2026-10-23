import re
import threading
import time

import cv2

from anpr.sources import frames, parse_source


def mask(text):
    # never send camera passwords to the browser: rtsp://user:pass@ip -> rtsp://***@ip
    return re.sub(r'//[^/@\s]+@', '//***@', str(text))


class Camera:
    """Runs the pipeline on one source in a background thread and keeps the latest annotated JPEG."""

    def __init__(self, pipeline, source):
        self.pipeline = pipeline
        self.source = parse_source(source)
        self.gen = 0  # bumped on every source change; the worker restarts when it sees a new value
        self.jpeg = None
        self.frame_id = 0
        self.fps = 0.0
        self.running = False
        self.error = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def set_source(self, source):
        self.source = parse_source(source)
        self.gen += 1

    def status(self):
        plates = [{'car_id': car_id, 'text': text, 'score': round(score, 2), 'reads': reads, 'time': seen}
                  for car_id, (text, score, reads, seen) in list(self.pipeline.best_plate.items())]
        plates.sort(key=lambda p: p['time'], reverse=True)
        return {'source': mask(self.source), 'running': self.running, 'error': self.error,
                'fps': round(self.fps, 1), 'plates': plates}

    def _run(self):
        while True:
            gen, source = self.gen, self.source
            self.pipeline.reset()
            self.running, self.error, self.fps = True, None, 0.0
            try:
                last = time.time()
                for frame in frames(source):
                    if self.gen != gen:
                        break
                    _, jpeg = cv2.imencode('.jpg', self.pipeline.process(frame))
                    now = time.time()
                    self.jpeg, self.frame_id = jpeg.tobytes(), self.frame_id + 1
                    self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
                    last = now
            except Exception as e:
                self.error = mask(e)
            self.running = False
            while self.gen == gen:  # video ended or source failed: wait for the next source
                time.sleep(0.2)
