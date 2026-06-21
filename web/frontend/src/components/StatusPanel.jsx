const stageNames = {
  intro: "开场邀请",
  color: "第一幕 · 择色",
  draw: "第二幕 · 筑景",
  spirit: "第三幕 · 唤灵",
  postcard: "第四幕 · 成色",
};

function Row({ label, children }) {
  return (
    <div className="border-b border-white/5 py-2 last:border-0">
      <dt className="text-xs text-white/40">{label}</dt>
      <dd className="mt-1 text-sm text-white/80">{children || "等待中"}</dd>
    </div>
  );
}

export default function StatusPanel({
  currentStage,
  color,
  colorSource,
  gesture,
  objectResult,
  matchedCharacter,
  isAutoAdvancing,
}) {
  return (
    <dl>
      <Row label="当前阶段">{stageNames[currentStage]}</Row>
      <Row label="流程状态">{isAutoAdvancing ? "正在自动推进" : "等待交互"}</Row>
      <Row label="当前颜色">
        {color && (
          <span className="inline-flex items-center gap-2">
            <span className="size-3 rounded-full border border-white/30" style={{ background: color.hex }} />
            {color.name}
          </span>
        )}
      </Row>
      <Row label="颜色来源">{colorSource}</Row>
      <Row label="当前手势">{gesture}</Row>
      <Row label="当前物象">{objectResult?.name}</Row>
      <Row label="自动匹配人物">{matchedCharacter?.name}</Row>
      <Row label="颜色含义">{color?.meaning}</Row>
    </dl>
  );
}
