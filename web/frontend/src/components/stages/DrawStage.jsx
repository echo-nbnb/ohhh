import DrawingCanvas from "../DrawingCanvas";

export default function DrawStage({
  mode,
  color,
  points,
  objectResult,
  onPointsChange,
  onClear,
  onComplete,
  isAutoAdvancing,
}) {
  return (
    <section className="stage-shell">
      <div className="w-full">
        <div className="text-center">
          <p className="eyebrow">STAGE 02 · DRAW</p>
          <h1 className="stage-title">第二幕 · 筑景</h1>
          <p className="stage-subtitle">我读懂你画下的湖大</p>
          <p className="mt-5 text-sm leading-7 text-white/55">
            伸出食指，开始画。<br />
            握拳结束绘画。张开手取消重画。
          </p>
        </div>

        <div className="mx-auto mt-8 max-w-5xl">
          <DrawingCanvas
            mode={mode}
            active={!isAutoAdvancing}
            color={color}
            points={points}
            onPointsChange={onPointsChange}
            onClear={onClear}
            onRecognize={onComplete}
          />
        </div>

        {objectResult && (
          <div className="mx-auto mt-6 max-w-xl rounded-xl border border-gold/25 bg-gold/5 px-6 py-5 text-center">
            <p className="whitespace-pre-line text-base leading-8 text-paper/85">{objectResult.narrative}</p>
            <p className="mt-3 text-xs text-white/35">即将进入唤灵</p>
          </div>
        )}
      </div>
    </section>
  );
}
