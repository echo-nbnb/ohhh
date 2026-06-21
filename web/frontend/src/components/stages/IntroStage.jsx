export default function IntroStage({ onStart }) {
  return (
    <section className="stage-shell min-h-[620px] text-center">
      <div className="mx-auto max-w-3xl">
        <p className="eyebrow">AI CULTURAL INTERACTION SYSTEM</p>
        <h1 className="mt-5 text-4xl font-semibold tracking-[0.2em] text-paper md:text-6xl">
          寻麓千年色
        </h1>
        <p className="mx-auto mt-8 max-w-2xl text-base leading-8 text-white/60 md:text-lg">
          这是一个关于湖大 / 湖湘文化的 AI 交互体验系统。
          <br />
          你将通过颜色、手势绘画与 AI 叙事，寻找属于自己的湖大千年色。
        </p>

        <div className="mx-auto mt-12 max-w-xl border-y border-gold/20 py-9 text-lg leading-9 text-paper/85">
          <p>岳麓千年，色隐其中。</p>
          <p>后来者，汝心何色？</p>
          <p className="mt-5 text-sm text-white/45">——无名学人，1170年</p>
        </div>

        <button className="button-primary mt-12 px-8 py-3 text-base" onClick={onStart}>
          开始寻色
        </button>
      </div>
    </section>
  );
}
