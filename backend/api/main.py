"""
usage (from backend/): uvicorn api.main:app
The initial source is CCTV_URL (default 0 = webcam); change it from the page.

Bind to 127.0.0.1 only (uvicorn's default): a source can be any local file or URL, so this must not be
reachable from the network.
"""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from anpr.pipeline import Pipeline

from .camera import Camera

FRONTEND_DIST = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'


@asynccontextmanager
async def lifespan(app):
    app.state.camera = Camera(Pipeline(), os.environ.get('CCTV_URL', '0'))
    app.state.camera.start()
    yield


app = FastAPI(title='ANPR', lifespan=lifespan)


class SourceIn(BaseModel):
    source: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@app.get('/api/status')
def status(request: Request):
    return request.app.state.camera.status()


@app.post('/api/source')
def set_source(body: SourceIn, request: Request):
    request.app.state.camera.set_source(body.source)
    return {}


@app.get('/api/stream')
async def stream(request: Request):
    """MJPEG: the browser shows each JPEG part as it arrives, so a plain <img> plays the live video."""
    camera = request.app.state.camera

    async def parts():
        seen = -1
        while True:
            if camera.jpeg is None or camera.frame_id == seen:
                await asyncio.sleep(0.02)
                continue
            seen, jpeg = camera.frame_id, camera.jpeg
            yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg) + jpeg + b'\r\n'

    return StreamingResponse(parts(), media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control': 'no-store'})


# production: serve the built React app from the same origin (npm run build in frontend/)
if FRONTEND_DIST.is_dir():
    app.mount('/', StaticFiles(directory=FRONTEND_DIST, html=True), name='frontend')
