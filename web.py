import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import live

# usage: python web.py [source]   then open http://127.0.0.1:8000
# source is the same as for live.py: 0 for a webcam, a video file, or an rtsp:// / http:// stream (default: CCTV_URL or 0)
PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web.html'), 'rb').read()

state = {'source': live.parse_source(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('CCTV_URL', '0')),
         'gen': 0, 'jpeg': None, 'frame_id': 0, 'fps': 0.0, 'running': False, 'error': None}


def mask(text):
    # never send camera passwords to the browser: rtsp://user:pass@ip -> rtsp://***@ip
    return re.sub(r'//[^/@\s]+@', '//***@', str(text))


def worker():
    # runs the detection loop; restarts whenever a new source is chosen in the page
    while True:
        gen, source = state['gen'], state['source']
        live.reset()
        state.update(running=True, error=None, fps=0.0)
        try:
            last = time.time()
            for frame in live.frames(source):
                if state['gen'] != gen:
                    break
                _, jpeg = cv2.imencode('.jpg', live.process(frame))
                now = time.time()
                state.update(jpeg=jpeg.tobytes(), frame_id=state['frame_id'] + 1,
                             fps=0.9 * state['fps'] + 0.1 / max(now - last, 1e-6))
                last = now
        except Exception as e:
            state['error'] = mask(e)
        state['running'] = False
        while state['gen'] == gen:  # video ended or source failed: wait for the next source
            time.sleep(0.2)


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.reply(200, 'text/html; charset=utf-8', PAGE)
        elif self.path == '/status':
            plates = [{'car_id': car_id, 'text': text, 'score': round(score, 2), 'time': seen}
                      for car_id, (text, score, seen) in list(live.best_plate.items())]
            plates.sort(key=lambda p: p['time'], reverse=True)
            self.reply(200, 'application/json', json.dumps({
                'source': mask(state['source']), 'running': state['running'], 'error': state['error'],
                'fps': round(state['fps'], 1), 'plates': plates}).encode())
        elif self.path == '/stream':
            # MJPEG: the browser shows each JPEG part as it arrives, so a plain <img> plays the live video
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            seen = -1
            try:
                while True:
                    if state['jpeg'] is None or state['frame_id'] == seen:
                        time.sleep(0.02)
                        continue
                    seen, jpeg = state['frame_id'], state['jpeg']
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg)
                                     + jpeg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.reply(404, 'text/plain', b'not found')

    def do_POST(self):
        if self.path != '/source':
            return self.reply(404, 'text/plain', b'not found')
        try:
            source = str(json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0)))).get('source', '')).strip()
        except ValueError:
            source = ''
        if not source:
            return self.reply(400, 'application/json', b'{"error": "source is required"}')
        state.update(source=live.parse_source(source), gen=state['gen'] + 1)
        self.reply(200, 'application/json', b'{}')

    def reply(self, code, content_type, body):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the terminal for plate readings


if __name__ == '__main__':
    threading.Thread(target=worker, daemon=True).start()
    # localhost only: the page can open any file or URL as a source, so it must not be reachable from the network
    print('Open http://127.0.0.1:8000')
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
