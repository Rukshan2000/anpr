export default function StatusPill({ status, offline }) {
  const [kind, label] = offline ? ['error', 'Server offline']
    : !status ? ['', 'Connecting…']
    : status.error ? ['error', 'Error']
    : status.running ? ['live', 'Live']
    : ['ended', 'Stopped'];
  return (
    <span className={`pill ${kind}`}>
      <span className="dot" />
      <span>{label}</span>
    </span>
  );
}
