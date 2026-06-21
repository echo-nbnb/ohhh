export default function Postcard({ color, objectResult, character, narrative }) {
  if (!color || !objectResult || !character) {
    return (
      <section className="panel grid min-h-96 place-items-center text-sm text-white/40">
        完成择色、筑景与唤灵后，明信片将在这里生成。
      </section>
    );
  }

  const title = narrative?.title || `${color.name} · ${objectResult.name}之境`;
  const paragraphs = narrative?.paragraphs || [
    `系统从你带来的物件中提取出一抹“${color.name}”，它象征${color.meaning}。`,
    `你在画布上留下轨迹，湖大的文脉将它读作“${objectResult.name}”。`,
    `${character.name}穿越千年回应了你：${character.spiritLine}`,
  ];

  return (
    <section className="panel">
      <p className="eyebrow">YOUR MILLENNIUM COLOR</p>
      <div className="mt-4 overflow-hidden rounded-xl border border-[#8c7652]/45 bg-[#e7dec9] text-[#263029] shadow-2xl">
        <div className="grid min-h-[430px] md:grid-cols-[1.4fr_0.8fr]">
          <div className="p-7 md:p-9">
            <p className="text-xs tracking-[0.25em] text-[#725f40]">寻麓千年色 · 专属明信片</p>
            <h2 className="mt-4 text-3xl font-semibold">{title}</h2>
            <div className="mt-6 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-current/20 px-3 py-1">底色 · {color.name}</span>
              <span className="rounded-full border border-current/20 px-3 py-1">物象 · {objectResult.name}</span>
              <span className="rounded-full border border-current/20 px-3 py-1">回声 · {character.name}</span>
            </div>
            <div className="mt-7 space-y-3 text-sm leading-7">
              {paragraphs.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
            </div>
            <p className="mt-7 border-l-2 pl-4 text-sm italic" style={{ borderColor: color.hex }}>
              “{character.spiritLine}”
            </p>
          </div>
          <div className="relative flex flex-col justify-between border-l border-[#8c7652]/25 p-7" style={{ background: `${color.hex}20` }}>
            <div className="absolute right-6 top-6 size-20 rounded-full border-2 border-[#8a3d35]/55 text-center text-xs leading-[76px] text-[#8a3d35] rotate-[-9deg]">
              千年色章
            </div>
            <div className="mt-24">
              <p className="text-xs text-[#725f40]">绘制物象</p>
              <p className="mt-2 text-4xl">{objectResult.name}</p>
            </div>
            <div>
              <div className="grid size-24 grid-cols-5 gap-1 bg-[#263029] p-2" aria-label="二维码占位">
                {Array.from({ length: 25 }).map((_, index) => (
                  <span key={index} className={index % 3 === 0 || index % 7 === 0 ? "bg-[#e7dec9]" : "bg-transparent"} />
                ))}
              </div>
              <p className="mt-2 text-xs text-[#725f40]">扫码带走这次相遇</p>
            </div>
          </div>
        </div>
      </div>
      <p className="mt-4 text-center text-sm leading-7 text-white/55">
        这就是你寻到的千年色。它不是被系统生成的答案，而是你与湖大千年文脉相遇后，留下的一次回声。
      </p>
    </section>
  );
}
