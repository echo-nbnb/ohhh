import { useCallback, useEffect, useMemo, useState } from "react";
import "./Act1Entry.css";

import bgUrl from "../../assets/act1/act1-bg.svg";
import ribbonUrl from "../../assets/act1/ribbon.svg";
import invite1Url from "../../assets/act1/invite-1.svg";
import signatureUrl from "../../assets/act1/signature.svg";
import invite2Url from "../../assets/act1/invite-2.svg";
import iconUrl from "../../assets/act0/act0-icon.svg";
import DissolveOverlay from "./DissolveOverlay";

function random(min, max) {
  return Math.random() * (max - min) + min;
}

function createFloatingIcons(count = 6) {
  const edgeAreas = [
    { xMin: -5, xMax: 105, yMin: -8, yMax: 18 },
    { xMin: -5, xMax: 105, yMin: 78, yMax: 108 },
    { xMin: -8, xMax: 18, yMin: 0, yMax: 100 },
    { xMin: 82, xMax: 108, yMin: 0, yMax: 100 },
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

export default function Act1Entry({ switchDelay = 5000, dissolveDelay = 13000, onComplete, onSkip, dissolveOnCompleteDelay = 800 }) {
  const [isSecondText, setIsSecondText] = useState(false);
  const [isDissolving, setIsDissolving] = useState(false);
  const [phase, setPhase] = useState("entering"); // entering → dissolved → act2
  const floatingIcons = useMemo(() => createFloatingIcons(6), []);

  useEffect(() => {
    const t1 = window.setTimeout(() => setIsSecondText(true), switchDelay);
    const t2 = window.setTimeout(() => setIsDissolving(true), dissolveDelay);
    return () => { window.clearTimeout(t1); window.clearTimeout(t2); };
  }, [switchDelay, dissolveDelay]);

  // Handler for dissolve completion: white screen → brief hold → advance
  const handleDissolveDone = useCallback(() => {
    setPhase("dissolved");
    if (onComplete) {
      setTimeout(() => onComplete(), dissolveOnCompleteDelay);
    }
  }, [onComplete, dissolveOnCompleteDelay]);

  const handleSkip = useCallback(() => {
    onSkip?.();
  }, [onSkip]);

  return (
    <section className={`act1${isDissolving ? " act1--fading" : ""}`} onClick={onSkip ? handleSkip : undefined} style={onSkip ? { cursor: "pointer" } : undefined}>
      {/* 静止底图 */}
      <img className="act1__bg" src={bgUrl} alt="" draggable="false" />

      {/* 轻微白色罩层，压住彩带，让它更融入底图 */}
      <div className="act1__veil" aria-hidden="true" />

      {/* 彩带：铺满全屏，整体 Y 轴缓慢呼吸 */}
      <img
        className="act1__ribbon"
        src={ribbonUrl}
        alt=""
        draggable="false"
      />

      {/* 邀请文字 1 */}
      {phase === "entering" && (
        <div
          className={[
            "act1__copy",
            "act1__invite",
            "act1__fade",
            !isSecondText ? "is-active" : "",
            isDissolving ? "is-dissolving" : "",
          ].join(" ")}
        >
          <img src={invite1Url} alt="邀请文字" draggable="false" />
        </div>
      )}

      {/* 邀请文字 2：5 秒后出现 */}
      {phase === "entering" && (
        <div
          className={[
            "act1__copy",
            "act1__invite",
            "act1__invite2",
            "act1__fade",
            isSecondText ? "is-active" : "",
            isDissolving ? "is-dissolving" : "",
          ].join(" ")}
        >
          <img src={invite2Url} alt="邀请文字二" draggable="false" />
        </div>
      )}

      {/* 落款：5 秒后消失 */}
      {phase === "entering" && (
        <div
          className={[
            "act1__copy",
            "act1__signature",
            isSecondText ? "is-hidden" : "",
            isDissolving ? "is-dissolving" : "",
          ].join(" ")}
        >
          <img src={signatureUrl} alt="落款" draggable="false" />
        </div>
      )}

      {/* 四周飘动 icon — 同第零幕逻辑 */}
      <div className="act1__icons" aria-hidden="true">
        {floatingIcons.map((item) => (
          <img
            key={item.id}
            className="act1__icon"
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

      {/* 碎光散开 → 白屏 */}
      <DissolveOverlay
        active={isDissolving}
        onDone={handleDissolveDone}
      />
    </section>
  );
}
