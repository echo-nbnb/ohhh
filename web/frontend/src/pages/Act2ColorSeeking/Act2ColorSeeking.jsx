import { useEffect, useMemo, useRef, useState } from "react";
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

const DEFAULT_COPY = {
  1: ["请将随身之物靠近光中。", "让它替你说话。"],
  2: ["让我看看……", "这件东西里", "藏着怎样的颜色。"],
  3: ["一缕明黄浮出光面，", "像旧纸上醒来的日色。"],
  4: ["明黄与深蓝相遇，", "像山门灯火照见夜色。"],
};

function hexToNorm(hex) {
  const clean = hex.replace("#", "");
  return [
    parseInt(clean.slice(0, 2), 16) / 255,
    parseInt(clean.slice(2, 4), 16) / 255,
    parseInt(clean.slice(4, 6), 16) / 255,
  ];
}

function blendColors(colors) {
  if (colors.length === 0) return [0.5, 0.5, 0.5];
  const norms = colors.map(hexToNorm);
  return [0, 1, 2].map(i =>
    norms.reduce((sum, c) => sum + c[i], 0) / norms.length
  );
}

export default function Act2ColorSeeking({
  step: controlledStep,
  recognizedColors = ["#F2C94C", "#2F5E9E"],
  copyByStep = DEFAULT_COPY,
  autoDemo = false,
  stepDuration = 5000,
  onComplete,
  completeDelay = 3000,
}) {
  const [internalStep, setInternalStep] = useState(1);
  const step = controlledStep ?? internalStep;

  useEffect(() => {
    if (!autoDemo || controlledStep !== undefined) return;
    const timer = window.setInterval(() => {
      setInternalStep((prev) => (prev >= 4 ? 1 : prev + 1));
    }, stepDuration);
    return () => window.clearInterval(timer);
  }, [autoDemo, controlledStep, stepDuration]);

  // Fire onComplete when step 4 is reached (both autoDemo and external control)
  const completeTimerRef = useRef(null);
  useEffect(() => {
    if (step === 4 && onComplete) {
      completeTimerRef.current = setTimeout(() => onComplete(), completeDelay);
    }
    return () => {
      if (completeTimerRef.current) clearTimeout(completeTimerRef.current);
    };
  }, [step, onComplete, completeDelay]);

  const diskColors = useMemo(() => {
    if (step === 3) return [recognizedColors[0]];
    if (step === 4) return [recognizedColors[0], recognizedColors[1]];
    return [];
  }, [step, recognizedColors]);

  const floatingIcons = useMemo(() => createFloatingIcons(6), []);
  const isColorActive = step >= 3;
  const currentCopy = copyByStep[step] || [];

  return (
    <section className="act2">
      <img className="act2__bg" src={bgUrl} alt="" draggable="false" />

      <div className="act2__center">
        <div className={["act2__disk", isColorActive ? "has-color" : "is-white"].join(" ")}>
          {isColorActive && diskColors.length > 0 && (
            <LiquidChrome
              colorA={hexToNorm(diskColors[0])}
              colorB={hexToNorm(diskColors.length >= 2 ? diskColors[1] : diskColors[0])}
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
      </div>

      <div className="act2__sideText" key={step}>
        {currentCopy.map((line, index) => (
          <div className="act2__sideTextLine" key={`${step}-${index}`}>
            {line}
          </div>
        ))}
      </div>

      {/* 四周飘动 icon — 同第零幕 */}
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
