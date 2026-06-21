import { useEffect, useMemo, useRef, useState } from "react";
import "./DissolveOverlay.css";

function random(min, max) {
  return Math.random() * (max - min) + min;
}

function createParticles(count = 60) {
  return Array.from({ length: count }).map((_, i) => ({
    id: i,
    x: random(0, 100),
    y: random(0, 100),
    size: random(3, 10),
    duration: random(1.5, 3.0),
    delay: random(0, 0.8),
    driftX: random(-150, 150),
    driftY: random(-220, -40),
    rotate: random(-90, 90),
    opacity: random(0.4, 1),
  }));
}

export default function DissolveOverlay({ active, onDone }) {
  const [stage, setStage] = useState("idle");
  const particles = useMemo(() => createParticles(80), []);

  useEffect(() => {
    if (active && stage === "idle") {
      setStage("dissolving");
      const maxDuration = 3.0 + 0.8 + 0.6; // longest particle + delay + white fade
      const timer = setTimeout(() => {
        setStage("done");
        onDone?.();
      }, maxDuration * 1000 + 200);
      return () => clearTimeout(timer);
    }
  }, [active, stage, onDone]);

  if (stage === "idle") return null;
  if (stage === "done") return <div className="dissolve-white" />;

  return (
    <div className="dissolve-overlay">
      {particles.map((p) => (
        <span
          key={p.id}
          className="dissolve-particle"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: `${p.size}px`,
            height: `${p.size}px`,
            "--drift-x": `${p.driftX}px`,
            "--drift-y": `${p.driftY}px`,
            "--rotate": `${p.rotate}deg`,
            "--duration": `${p.duration}s`,
            "--delay": `${p.delay}s`,
            opacity: p.opacity,
          }}
        />
      ))}
    </div>
  );
}
