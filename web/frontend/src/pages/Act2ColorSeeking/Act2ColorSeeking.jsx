import { useEffect, useMemo } from "react";
import "./Act2ColorSeeking.css";

import bgUrl from "../../assets/act2/act2-bg.svg";
import diskFrameUrl from "../../assets/act2/color-disk.svg";
import orbitRingUrl from "../../assets/act2/orbit-ring.svg";
import iconUrl from "../../assets/act0/act0-icon.svg";
import LiquidChrome from "./LiquidChrome";

function random(min, max) {
  return Math.random() * (max - min) + min;
}

function createFloatingIcons(count = 6) {
  const edgeAreas = [
    { xMin: -8, xMax: 15, yMin: 5, yMax: 95 },
    { xMin: 85, xMax: 108, yMin: 5, yMax: 95 },
  ];
  return Array.from({ length: count }).map((_, index) => {
    const area = edgeAreas[index % edgeAreas.length];
    return {
      id: index,
      x: random(area.xMin, area.xMax),
      y: random(area.yMin, area.yMax),
      size: random(36, 110),
      opacity: random(0.25, 0.60),
      rotate: random(0, 360),
      duration: random(10, 22),
      delay: random(-18, 0),
      driftX: random(-90, 90),
      driftY: random(-70, 70),
    };
  });
}

function hexToNorm(hex) {
  if (!hex || typeof hex !== "string") return [0.5, 0.5, 0.5];
  const clean = hex.replace("#", "");
  return [
    parseInt(clean.slice(0, 2), 16) / 255,
    parseInt(clean.slice(2, 4), 16) / 255,
    parseInt(clean.slice(4, 6), 16) / 255,
  ];
}

export default function Act2ColorSeeking({
  round = 1,                   // 当前轮次：1 or 2
  firstColor = null,           // {hex, name}
  secondColor = null,          // {hex, name}
  isDetecting = false,
  stableColorName = null,
  stableSeconds = 0,
  confirmSeconds = 3,
  onComplete,
  completeDelay = 2000,
}) {
  const isDone = secondColor !== null;  // 两色齐备

  // 两色完成 → 自动进入下一幕
  useEffect(() => {
    if (!isDone || !onComplete) return;
    const t = setTimeout(() => onComplete(), completeDelay);
    return () => clearTimeout(t);
  }, [isDone, onComplete, completeDelay]);

  const floatingIcons = useMemo(() => createFloatingIcons(6), []);

  // 颜色盘：第一色确认后显示第一色，两色齐备后双色混合
  const diskColors = useMemo(() => {
    if (secondColor?.hex) return [secondColor.hex, firstColor?.hex];
    if (firstColor?.hex) return [firstColor.hex];
    return [];
  }, [firstColor, secondColor]);

  const showColor = firstColor !== null;

  return (
    <section className="act2">
      <img className="act2__bg" src={bgUrl} alt="" draggable="false" />

      {/* ── 中央颜色盘 ── */}
      <div className="act2__center">
        <div className={["act2__disk", showColor ? "has-color" : "is-white"].join(" ")}>
          {showColor && diskColors.length > 0 && (
            <LiquidChrome
              colorA={hexToNorm(diskColors[0])}
              colorB={hexToNorm(diskColors[1] || diskColors[0])}
              speed={0.25}
              amplitude={0.40}
              frequencyX={2.0}
              frequencyY={1.6}
              interactive={false}
            />
          )}
        </div>
        <img className="act2__diskFrame" src={diskFrameUrl} alt="" draggable="false" />
        <img className="act2__orbitRing" src={orbitRingUrl} alt="" draggable="false" />

        {/* ── 检测框 ── */}
        <div className={[
          "act2__detectBox",
          isDetecting && "act2__detectBox--active",
          isDetecting && stableColorName && "act2__detectBox--stable",
          isDone && "act2__detectBox--done",
        ].filter(Boolean).join(" ")}>
          <div className="act2__detectBoxInner">
            {/* 四角 */}
            <span className="act2__detectCorner act2__detectCorner--tl" />
            <span className="act2__detectCorner act2__detectCorner--tr" />
            <span className="act2__detectCorner act2__detectCorner--bl" />
            <span className="act2__detectCorner act2__detectCorner--br" />

            {/* 轮次标记 */}
            {!isDone && (
              <div className="act2__roundBadge">{round}/2</div>
            )}

            {/* 已完成的颜色色块 */}
            {firstColor && (
              <div className="act2__swatchRow">
                <div className="act2__miniSwatch" style={{ background: firstColor.hex }}>
                  <span className="act2__miniSwatchCheck">✓</span>
                </div>
                <span className="act2__miniSwatchName">{firstColor.name}</span>
              </div>
            )}
            {secondColor && (
              <div className="act2__swatchRow act2__swatchRow--second">
                <div className="act2__miniSwatch" style={{ background: secondColor.hex }}>
                  <span className="act2__miniSwatchCheck">✓</span>
                </div>
                <span className="act2__miniSwatchName">{secondColor.name}</span>
              </div>
            )}

            {/* 检测进度 */}
            {isDetecting && (
              <div className="act2__detectProgress">
                {stableColorName ? (
                  <>
                    <span className="act2__detectLabel">{stableColorName}</span>
                    <span className="act2__detectTimer">
                      {Math.max(0, confirmSeconds - stableSeconds).toFixed(1)}s
                    </span>
                  </>
                ) : (
                  <span className="act2__detectLabel act2__detectLabel--dim">识别中……</span>
                )}
              </div>
            )}

            {/* 等待按 R */}
            {!isDetecting && !isDone && (
              <div className="act2__detectHint">
                {firstColor ? (
                  <>
                    <span className="act2__detectHintKey">R</span>
                    <span className="act2__detectHintText">再寻一色，按 R 键开始</span>
                  </>
                ) : (
                  <>
                    <span className="act2__detectHintKey">R</span>
                    <span className="act2__detectHintText">将物品放入框内，按 R 键开始</span>
                  </>
                )}
              </div>
            )}

            {/* 两色完成 */}
            {isDone && (
              <div className="act2__detectDone">
                <span className="act2__detectDoneText">两色相遇</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── 右侧文字 ── */}
      <div className="act2__sideText">
        {isDone ? (
          <>
            <div className="act2__sideTextLine">{firstColor?.name}与{secondColor?.name}相遇，</div>
            <div className="act2__sideTextLine">像山门灯火照见夜色。</div>
          </>
        ) : firstColor ? (
          <>
            <div className="act2__sideTextLine">一缕{firstColor.name}浮出光面，</div>
            <div className="act2__sideTextLine">再寻一色，让它们相遇。</div>
          </>
        ) : isDetecting ? (
          <>
            <div className="act2__sideTextLine">让我看看……</div>
            <div className="act2__sideTextLine">这件东西里藏着怎样的颜色。</div>
          </>
        ) : (
          <>
            <div className="act2__sideTextLine">请将随身之物靠近光中。</div>
            <div className="act2__sideTextLine">让它替你说话。</div>
          </>
        )}
      </div>

      {/* ── 四周飘动 icon ── */}
      <div className="act2__icons" aria-hidden="true">
        {floatingIcons.map((item) => (
          <img
            key={item.id}
            className="act2__icon"
            src={iconUrl}
            alt=""
            draggable="false"
            style={{
              left: `${item.x}%`,
              top: `${item.y}%`,
              width: `${item.size}px`,
              height: `${item.size}px`,
              opacity: item.opacity,
              transform: `rotate(${item.rotate}deg)`,
              "--duration": `${item.duration}s`,
              "--delay": `${item.delay}s`,
              "--drift-x": `${item.driftX}px`,
              "--drift-y": `${item.driftY}px`,
            }}
          />
        ))}
      </div>
    </section>
  );
}
