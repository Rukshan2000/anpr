import { useEffect, useState } from 'react';
import { getStatus, setSource } from './api.js';
import PlateList from './components/PlateList.jsx';
import SourceForm from './components/SourceForm.jsx';
import StatusPill from './components/StatusPill.jsx';
import VideoFeed from './components/VideoFeed.jsx';

export default function App() {
  const [status, setStatus] = useState(null);
  const [offline, setOffline] = useState(false);
  const [sourceError, setSourceError] = useState('');

  useEffect(() => {
    const poll = () => getStatus().then((s) => { setStatus(s); setOffline(false); }, () => setOffline(true));
    poll();
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, []);

  const changeSource = (source) => {
    setSourceError('');
    setSource(source).catch((e) => setSourceError(e.message));
  };

  return (
    <>
      <header>
        <h1>ANPR Live</h1>
        <StatusPill status={status} offline={offline} />
      </header>
      <main>
        <section className="card">
          <SourceForm onSubmit={changeSource} />
          <VideoFeed />
          <div className="meta">
            <span>Source: {status?.source ?? '–'}</span>
            {status?.running && <span>{status.fps} fps</span>}
          </div>
          <div className="error" role="alert">{sourceError || status?.error}</div>
        </section>
        <aside className="card">
          <PlateList plates={status?.plates ?? []} />
        </aside>
      </main>
    </>
  );
}
