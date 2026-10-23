export default function PlateList({ plates }) {
  return (
    <>
      <h2>Plates <span className="small">{plates.length ? `${plates.length} cars` : ''}</span></h2>
      {plates.length === 0 ? <p className="empty">No plates read yet.</p> : (
        <ol>
          {plates.map((p) => {
            const pct = Math.round(p.score * 100);
            return (
              <li key={p.car_id}>
                <span className="plate">{p.text}</span>
                <span className="small">
                  Car #{p.car_id} · {new Date(p.time * 1000).toLocaleTimeString()} · {pct}% · {p.reads} {p.reads === 1 ? 'read' : 'reads'}
                </span>
                <div className="bar"><span style={{ width: `${pct}%` }} /></div>
              </li>
            );
          })}
        </ol>
      )}
    </>
  );
}
