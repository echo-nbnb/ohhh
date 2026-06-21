import { useEffect, useMemo } from "react";
import "./Act0.css";

import bgUrl from "../../assets/act0/act0-bg.svg";
import titleUrl from "../../assets/act0/act0-title.svg";
import iconUrl from "../../assets/act0/act0-icon.svg";
import lightUrl from "../../assets/act0/act0-light.svg";

function random(min, max) {
  return Math.random() * (max - min) + min;
}

function createFloatingIcons(count = 22) {
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

function createLightLayers(count = 5) {
  return Array.from({ length: count }).map((_, index) => ({
    id: index,
    opacity: random(0.50, 0.85),
    duration: random(14, 24),
    delay: random(-8, 0),
  }));
}

export default function Act0({ onNext, autoAdvanceDelay = 4000 }) {
  const floatingIcons = useMemo(() => createFloatingIcons(6), []);
  const lightLayers = useMemo(() => createLightLayers(6), []);

  // Auto-advance after delay
  useEffect(() => {
    if (!onNext) return;
    const timer = setTimeout(() => onNext(), autoAdvanceDelay);
    return () => clearTimeout(timer);
  }, [onNext, autoAdvanceDelay]);

  // Click on center area to skip
  const handleClick = () => onNext?.();

  return (
    <section className="act0">
      {/* 静止背景 */}
      <img className="act0__bg" src={bgUrl} alt="" draggable="false" />

      {/* 透明光效层 */}
      <div className="act0__lights" aria-hidden="true">
        {lightLayers.map((item) => (
          <img
            key={item.id}
            className="act0__light"
            src={lightUrl}
            alt=""
            draggable="false"
            style={{
              opacity: item.opacity,
              "--duration": `${item.duration}s`,
              "--delay": `${item.delay}s`,
            }}
          />
        ))}
      </div>

      {/* 四周飘动 icon */}
      <div className="act0__icons" aria-hidden="true">
        {floatingIcons.map((item) => (
          <img
            key={item.id}
            className="act0__icon"
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

      {/* 中央标题 */}
      <main className="act0__center" onClick={handleClick} style={{ cursor: onNext ? "pointer" : "default" }}>
        <img
          className="act0__title"
          src={titleUrl}
          alt="寻麓千年色 · 点击开始"
          draggable="false"
        />
      </main>
    </section>
  );
}
