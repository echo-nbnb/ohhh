export default function SpiritMonologue({ lines, visibleCount }) {
  return (
    <div className="spirit-module-monologue">
      {lines.slice(0, visibleCount).map((line, index) => (
        <p
          key={`${index}-${line}`}
          className="spirit-module-monologue-line"
          style={{ paddingLeft: Math.min(index, 3) * 12 }}
        >
          {line}
        </p>
      ))}
    </div>
  );
}
