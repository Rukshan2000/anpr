import { useState } from 'react';
import { STREAM_URL } from '../api.js';

export default function VideoFeed() {
  const [src, setSrc] = useState(STREAM_URL);
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="video">
      <img src={src} alt="Live video with detected vehicles and plates" onLoad={() => setLoaded(true)}
           // if the server restarts, reconnect the stream
           onError={() => setTimeout(() => setSrc(`${STREAM_URL}?${Date.now()}`), 2000)} />
      {!loaded && <p>Waiting for video…</p>}
    </div>
  );
}
