export default function ColorStage({ mode, color, colorSource, onDetect, isAutoAdvancing }) {
  return (
    <section className="stage-shell min-h-[620px]">
      <div className="mx-auto w-full max-w-3xl text-center">
        <p className="eyebrow">STAGE 01 · COLOR</p>
        <h1 className="stage-title">第一幕 · 择色</h1>
        <p className="stage-subtitle">我从你身上看见一种颜色</p>

        <div className="mx-auto mt-10 max-w-xl text-base leading-8 text-white/65">
          <p>把一件你愿意带来的东西放下吧。</p>
          <p>不必贵重。</p>
          <p>它只需要属于你。</p>
        </div>

        {!color ? (
          <div className="mt-12">
            {mode === "demo" ? (
              <button
                className="button-primary px-8 py-3 text-base"
                onClick={onDetect}
                disabled={isAutoAdvancing}
              >
                {isAutoAdvancing ? "正在识别……" : "开始识别颜色"}
              </button>
            ) : (
              <div className="inline-flex items-center gap-3 rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm text-white/55">
                <span className="status-dot status-connecting" />
                等待后端识别物件颜色
              </div>
            )}
          </div>
        ) : (
          <div className="mx-auto mt-12 max-w-2xl rounded-2xl border border-gold/25 bg-black/20 p-7">
            <span
              className="mx-auto block size-20 rounded-full border-4 border-white/15 shadow-xl"
              style={{ background: color.hex }}
            />
            <p className="mt-6 text-sm leading-7 text-white/70">
              我看到了。
              <br />
              你的底色，是<span className="mx-1 text-xl text-paper">【{color.name}】</span>。
              <br />
              它不是表面的颜色，
              <br />
              而是你今天带来的心绪。
            </p>
            <blockquote className="mt-5 text-sm leading-7 text-gold/85">“{color.voiceLine}”</blockquote>
            <p className="mt-4 text-xs text-white/35">{colorSource} · 即将进入筑景</p>
          </div>
        )}
      </div>
    </section>
  );
}
