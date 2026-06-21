export default function SystemLog({ logs }) {
  return (
    <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
      {logs.length === 0 && <p className="text-sm text-white/35">尚无系统事件。</p>}
      {[...logs].reverse().map((log) => (
        <div key={log.id} className="rounded-md border border-white/5 bg-black/15 px-3 py-2 text-xs">
          <span className="text-gold/70">{log.time}</span>
          <p className="mt-1 text-white/65">{log.message}</p>
        </div>
      ))}
    </div>
  );
}
