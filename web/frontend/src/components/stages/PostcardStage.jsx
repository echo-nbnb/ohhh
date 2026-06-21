import Postcard from "../Postcard";

export default function PostcardStage({ color, objectResult, character, narrative, onRestart }) {
  return (
    <section className="stage-shell">
      <div className="w-full">
        <div className="text-center">
          <p className="eyebrow">STAGE 04 · POSTCARD</p>
          <h1 className="stage-title">第四幕 · 成色</h1>
          <p className="stage-subtitle">带走属于你的千年色</p>
          <p className="mt-6 text-lg text-paper/80">这是你的千年色。</p>
        </div>

        <div className="mx-auto mt-8 max-w-5xl">
          <Postcard
            color={color}
            objectResult={objectResult}
            character={character}
            narrative={narrative}
          />
        </div>

        <div className="mt-8 text-center">
          <button className="button-secondary px-7 py-3" onClick={onRestart}>重新开始</button>
        </div>
      </div>
    </section>
  );
}
