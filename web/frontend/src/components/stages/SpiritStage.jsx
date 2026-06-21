import SpiritReveal from "../SpiritReveal";

export default function SpiritStage({ status, character, onEnterPostcard }) {
  return (
    <section className="stage-shell min-h-[620px]">
      <div className="w-full text-center">
        <p className="eyebrow">STAGE 03 · SPIRIT</p>
        <h1 className="stage-title">第三幕 · 唤灵</h1>
        <p className="stage-subtitle">千年中有人回应了你</p>
        <p className="mx-auto mt-7 max-w-2xl text-sm leading-7 text-white/55">
          颜色已经展开。物象也已经落下。
          <br />
          现在，我要在千年的文脉里，寻找一个与你相遇的人。
        </p>
        <div className="mx-auto mt-8 max-w-3xl">
          <SpiritReveal status={status} character={character} onEnterPostcard={onEnterPostcard} />
        </div>
      </div>
    </section>
  );
}
