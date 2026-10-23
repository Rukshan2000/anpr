import { useState } from 'react';

// paths are relative to backend/, where the API runs
const PRESETS = [
  ['Webcam', '0'],
  ['Sample video', 'data/sample.mp4'],
  ['TfL Piccadilly Circus', 'https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.07450.mp4'],
];

export default function SourceForm({ onSubmit }) {
  const [value, setValue] = useState('');

  const submit = (source) => {
    setValue(source);
    if (source.trim()) onSubmit(source.trim());
  };

  return (
    <>
      <form onSubmit={(e) => { e.preventDefault(); submit(value); }}>
        <label htmlFor="source">Video source</label>
        <input id="source" value={value} onChange={(e) => setValue(e.target.value)} autoComplete="off"
               spellCheck="false" placeholder="0 for webcam, data/sample.mp4, or rtsp://user:pass@camera-ip:554/stream" />
        <button className="primary" type="submit">Connect</button>
      </form>
      <div className="presets">
        {PRESETS.map(([label, source]) => (
          <button key={source} type="button" onClick={() => submit(source)}>{label}</button>
        ))}
      </div>
    </>
  );
}
