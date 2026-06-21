const statusCopy = {
  idle: "等待进入唤灵",
  searching: "正在寻找回应你的人……",
  found: "找到了。",
};

export default function SpiritReveal({ status, character, onEnterPostcard }) {
  const speaking = status === "speaking";
  const revealed = status === "revealed";

  return (
    <div className="min-h-72 rounded-xl border border-white/10 bg-black/20 p-5">
      <p className="eyebrow">SPIRIT REVEAL</p>
      <div className="mt-5 text-center">
        {(status === "idle" || status === "searching" || status === "found") && (
          <>
            <p className="text-sm text-white/55">{statusCopy[status]}</p>
            {status === "found" && <p className="mt-8 text-4xl tracking-[0.2em] text-paper">????</p>}
            {status === "found" && <p className="mt-5 text-sm text-white/45">他还不能告诉你名字。你要先听他说完。</p>}
          </>
        )}

        {speaking && character && (
          <div>
            <p className="text-3xl tracking-[0.2em] text-paper">????</p>
            <div className="mx-auto mt-6 max-w-sm space-y-2 text-sm leading-7 text-white/75">
              {character.monologue.map((line) => <p key={line}>{line}</p>)}
            </div>
          </div>
        )}

        {revealed && character && (
          <div>
            <p className="text-xs tracking-[0.25em] text-gold">刚才与你说话的，是他。</p>
            <h3 className="mt-3 text-3xl text-paper">{character.name}</h3>
            <p className="mt-2 text-sm text-white/55">{character.title}</p>
            <p className="mx-auto mt-5 max-w-sm text-sm leading-7 text-white/70">{character.reason}</p>
            <blockquote className="mx-auto mt-5 max-w-sm border-l-2 border-gold/60 pl-4 text-left text-sm leading-7 text-paper/80">
              {character.spiritLine}
            </blockquote>
            <button className="button-primary mt-6" onClick={onEnterPostcard}>进入成色</button>
          </div>
        )}
      </div>
    </div>
  );
}
