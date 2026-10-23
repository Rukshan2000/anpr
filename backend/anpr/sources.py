import os
import threading
import time

import cv2

# CCTV cameras drop packets over UDP, which smears frames
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'


def parse_source(text):
    """'0' -> webcam 0; anything else is a file path or stream URL."""
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
