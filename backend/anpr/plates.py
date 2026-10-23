import functools
import string
from collections import defaultdict

import cv2

# OCR confuses these; fix them up by position, since the plate format says whether a letter or digit belongs there
CHAR_TO_INT = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

# UK format AA00AAA: letters everywhere except positions 2 and 3
PLATE_LENGTH = 7
LETTER_POSITIONS = {0, 1, 4, 5, 6}

ALLOWLIST = string.ascii_uppercase + string.digits
MIN_TEXT_SCORE = 0.3  # below this EasyOCR is mostly guessing
MIN_CROP_HEIGHT = 64  # EasyOCR misses characters on tiny crops, so upscale small plates to at least this


@functools.cache
def reader():
    import easyocr  # slow import (torch), only pay for it when a plate is actually read
    return easyocr.Reader(['en'], gpu=False)


def license_complies_format(text):
    return len(text) == PLATE_LENGTH and all(
        (c in string.ascii_uppercase or c in INT_TO_CHAR) if i in LETTER_POSITIONS else (c in string.digits or c in CHAR_TO_INT)
        for i, c in enumerate(text))


def format_license(text):
    return ''.join((INT_TO_CHAR if i in LETTER_POSITIONS else CHAR_TO_INT).get(c, c) for i, c in enumerate(text))


def read_license_plate(crop):
    """Read the plate text from a BGR crop. Returns (text, score), or (None, None) if nothing plate-like was read."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < MIN_CROP_HEIGHT:
        scale = MIN_CROP_HEIGHT / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for _, text, score in reader().readtext(gray, allowlist=ALLOWLIST):
        text = text.upper().replace(' ', '')
        if score >= MIN_TEXT_SCORE and license_complies_format(text):
            return format_license(text), score
    return None, None


def get_car(plate_bbox, tracks):
    """Return the tracked car (x1, y1, x2, y2, car_id) whose box contains the plate's centre, or None."""
    x1, y1, x2, y2 = plate_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for track in tracks:
        xcar1, ycar1, xcar2, ycar2, _ = track
        if xcar1 < cx < xcar2 and ycar1 < cy < ycar2:
            return track
    return None


class PlateVotes:
    """
    Combine all reads of one car: every character position votes, weighted by OCR score, so one bad frame
    can't flip a character and the plate converges as the car stays in view.
    """

    def __init__(self):
        self.votes = [defaultdict(float) for _ in range(PLATE_LENGTH)]
        self.reads = 0

    def add(self, text, score):
        self.reads += 1
        for position, c in zip(self.votes, text):
            position[c] += score

    def text(self):
        return ''.join(max(position, key=position.get) for position in self.votes)

    def score(self):
        """Mean OCR score behind the winning characters; reads that disagree pull it down."""
        return sum(max(position.values()) for position in self.votes) / (PLATE_LENGTH * self.reads)
