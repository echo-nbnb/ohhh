const stages = [
  ["intro", "开场邀请"],
  ["color", "择色"],
  ["draw", "筑景"],
  ["spirit", "唤灵"],
  ["postcard", "成色"],
];

export default function StageStepper({ currentStage }) {
  const activeIndex = stages.findIndex(([id]) => id === currentStage);
  return (
    <ol className="mx-auto flex w-full max-w-4xl items-start justify-between">
      {stages.map(([id, label], index) => (
        <li
          key={id}
          className={`relative flex flex-1 flex-col items-center gap-2 text-center text-xs ${
            index === activeIndex
              ? "text-paper"
              : index < activeIndex
                ? "text-gold/70"
                : "text-white/30"
          }`}
        >
          {index > 0 && (
            <span className={`absolute right-1/2 top-3 h-px w-full ${index <= activeIndex ? "bg-gold/45" : "bg-white/10"}`} />
          )}
          <span className={`relative z-10 flex size-7 items-center justify-center rounded-full border bg-ink ${
            index === activeIndex ? "border-gold bg-gold text-ink" : "border-current"
          }`}>
            {String(index + 1).padStart(2, "0")}
          </span>
          <span>{label}</span>
        </li>
      ))}
    </ol>
  );
}
