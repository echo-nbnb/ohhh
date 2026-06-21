import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./Act3FormingVision.css";

import bgUrl from "../../assets/act3/act3-bg.svg";
import titleUrl from "../../assets/act3/act3-title.svg";
import barTopUrl from "../../assets/act3/bar-top.svg";
import barTopSoftUrl from "../../assets/act3/bar-top-soft.svg";
import barBottomSoftUrl from "../../assets/act3/bar-bottom-soft.svg";
import barBottomUrl from "../../assets/act3/bar-bottom.svg";
import barGradientUrl from "../../assets/act3/bar-gradient.svg";
import bridgeDotsUrl from "../../assets/act3/bridge-dots.svg";
import iconUrl from "../../assets/act0/act0-icon.svg";

function random(min, max) { return Math.random() * (max - min) + min; }
function createFloatingIcons(count = 6) {
  const edgeAreas = [
    { xMin: -8, xMax: 15, yMin: 5, yMax: 95 },
    { xMin: 85, xMax: 108, yMin: 5, yMax: 95 },
    { xMin: -8, xMax: 15, yMin: 5, yMax: 95 },
    { xMin: 85, xMax: 108, yMin: 5, yMax: 95 },
  ];
  return Array.from({ length: count }).map((_, i) => {
    const a = edgeAreas[i % edgeAreas.length];
    return { id: i, x: random(a.xMin, a.xMax), y: random(a.yMin, a.yMax), size: random(36, 110), opacity: random(0.25, 0.60), rotate: random(0, 360), duration: random(10, 22), delay: random(-18, 0), driftX: random(-90, 90), driftY: random(-70, 70) };
  });
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function createMaskStyle(maskUrl, bg, op = 1) {
  return { background: bg, opacity: op, WebkitMaskImage: `url(${maskUrl})`, maskImage: `url(${maskUrl})`, WebkitMaskRepeat: "no-repeat", maskRepeat: "no-repeat", WebkitMaskSize: "100% 100%", maskSize: "100% 100%", WebkitMaskPosition: "center", maskPosition: "center" };
}

function drawStroke(ctx, stroke, opts = {}) {
  const pts = stroke.points || [];
  if (pts.length < 2) return;
  ctx.save(); ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.strokeStyle = opts.strokeStyle || "rgba(55,55,55,0.82)";
  ctx.lineWidth = opts.lineWidth || 4;
  ctx.shadowColor = opts.shadowColor || "rgba(255,255,255,0.55)";
  ctx.shadowBlur = opts.shadowBlur || 2;
  ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke(); ctx.restore();
}

function getSketchBounds(strokes, w, h, pad = 34) {
  const pts = strokes.flatMap((s) => s.points || []);
  if (!pts.length) return null;
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  pts.forEach((p) => { x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y); x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y); });
  x1 = clamp(x1 - pad, 0, w); y1 = clamp(y1 - pad, 0, h); x2 = clamp(x2 + pad, 0, w); y2 = clamp(y2 + pad, 0, h);
  const bw = Math.max(1, x2 - x1), bh = Math.max(1, y2 - y1), cx = x1 + bw / 2, cy = y1 + bh / 2;
  return { x: x1, y: y1, width: bw, height: bh, centerX: cx, centerY: cy, normalized: { x: x1 / w, y: y1 / h, width: bw / w, height: bh / h, centerX: cx / w, centerY: cy / h } };
}

function exportSketchImageData(strokes, w, h) {
  const c = document.createElement("canvas"); c.width = Math.max(1, Math.floor(w)); c.height = Math.max(1, Math.floor(h));
  const ctx = c.getContext("2d");
  strokes.forEach((s) => drawStroke(ctx, s, { strokeStyle: "rgba(0,0,0,0.95)", lineWidth: 6, shadowBlur: 0, shadowColor: "transparent" }));
  return c.toDataURL("image/png");
}

function resolveObjectImage(label) {
  try {
    // 命名约定：assets/act3/objects/{物象名}.svg
    return new URL(`../../assets/act3/objects/${label}.svg`, import.meta.url).href;
  } catch {
    return bridgeDotsUrl; // 兜底
  }
}

async function mockRecognizeSketch() {
  await new Promise((r) => setTimeout(r, 1100));
  return { label: "桥", description: ["你画下了一座桥。", "桥连接两岸，也连接出发与归来。"], overlay: { scale: 1.35, offsetX: 0, offsetY: 0 } };
}

export default function Act3FormingVision({ primaryColor = "#F2E700", secondaryColor = "#355BFF", maxRounds = 2, onRecognizeSketch, onComplete, completeDelay = 2000, remotePoints = [] }) {
  console.log("[Act3] Mounting with primaryColor:", primaryColor, "secondaryColor:", secondaryColor);
  const sceneRef = useRef(null), canvasRef = useRef(null);
  const [sceneSize, setSceneSize] = useState({ width: 1, height: 1 });
  const [round, setRound] = useState(1);
  const [drawingEnabled, setDrawingEnabled] = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [sideLines, setSideLines] = useState(["颜色已经出现了。", "但它还没有形状。"]);
  const [centerLines, setCenterLines] = useState([]);
  const [confirmedItems, setConfirmedItems] = useState([]);
  const [currentStrokes, setCurrentStrokes] = useState([]);
  const [activeStroke, setActiveStroke] = useState(null);
  const floatingIcons = useMemo(() => createFloatingIcons(6), []);
  const allCurrentStrokes = useMemo(() => [...currentStrokes, ...(activeStroke ? [activeStroke] : [])], [currentStrokes, activeStroke]);

  useEffect(() => {
    const ts = [];
    setSideLines(["颜色已经出现了。", "但它还没有形状。"]);
    setCenterLines([]);
    ts.push(setTimeout(() => setCenterLines(["伸出食指，开始作画。"]), 1800));
    ts.push(setTimeout(() => { setSideLines(["请画下此刻你想到的一个意象。"]); setCenterLines(["一棵树、一条河、一座山、一盏灯、一只鸟，", "或任何你想留下的形状。", "画完后点击右下角确认。"]); setDrawingEnabled(true); }, 4800));
    return () => ts.forEach(clearTimeout);
  }, []);

  useEffect(() => {
    const resize = () => {
      if (!sceneRef.current || !canvasRef.current) return;
      const r = sceneRef.current.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
      setSceneSize({ width: r.width, height: r.height });
      const c = canvasRef.current; c.width = Math.max(1, Math.floor(r.width * dpr)); c.height = Math.max(1, Math.floor(r.height * dpr));
      c.style.width = `${r.width}px`; c.style.height = `${r.height}px`;
      c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize(); window.addEventListener("resize", resize); return () => window.removeEventListener("resize", resize);
  }, []);

  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext("2d"); ctx.clearRect(0, 0, sceneSize.width, sceneSize.height);
    confirmedItems.forEach((it) => it.strokes.forEach((s) => drawStroke(ctx, s, { strokeStyle: "rgba(72,72,72,0.58)", lineWidth: 4, shadowBlur: 1 })));
    currentStrokes.forEach((s) => drawStroke(ctx, s, { strokeStyle: "rgba(50,50,50,0.78)", lineWidth: 4 }));
    if (activeStroke) drawStroke(ctx, activeStroke, { strokeStyle: "rgba(35,35,35,0.92)", lineWidth: 4 });
    // Render remote points (from backend/camera) as faint glowing dots
    if (remotePoints.length > 1) {
      ctx.save();
      ctx.strokeStyle = "rgba(180,160,120,0.45)";
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowColor = "rgba(200,180,140,0.4)";
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(remotePoints[0].x, remotePoints[0].y);
      for (let i = 1; i < remotePoints.length; i++) ctx.lineTo(remotePoints[i].x, remotePoints[i].y);
      ctx.stroke();
      ctx.restore();
    }
  }, [sceneSize, confirmedItems, currentStrokes, activeStroke, remotePoints]);

  const getRel = useCallback((cx, cy) => { const r = sceneRef.current.getBoundingClientRect(); return { x: clamp(cx - r.left, 0, r.width), y: clamp(cy - r.top, 0, r.height), t: Date.now() }; }, []);

  const handleDown = useCallback((e) => {
    if (!drawingEnabled || isRecognizing) return;
    if (e.target.closest(".act3__toolbar")) return;
    e.preventDefault(); e.currentTarget.setPointerCapture?.(e.pointerId);
    setActiveStroke({ id: `${Date.now()}-${Math.random()}`, round, points: [getRel(e.clientX, e.clientY)] });
  }, [drawingEnabled, isRecognizing, getRel, round]);

  const handleMove = useCallback((e) => {
    if (!activeStroke || !drawingEnabled || isRecognizing) return;
    e.preventDefault();
    const p = getRel(e.clientX, e.clientY);
    setActiveStroke((prev) => { if (!prev) return prev; const last = prev.points[prev.points.length - 1]; if (Math.hypot(p.x - last.x, p.y - last.y) < 2) return prev; return { ...prev, points: [...prev.points, p] }; });
  }, [activeStroke, drawingEnabled, isRecognizing, getRel]);

  const finish = useCallback(() => { if (!activeStroke) return; if ((activeStroke.points || []).length > 1) setCurrentStrokes((p) => [...p, activeStroke]); setActiveStroke(null); }, [activeStroke]);
  useEffect(() => { window.addEventListener("pointerup", finish); window.addEventListener("pointercancel", finish); return () => { window.removeEventListener("pointerup", finish); window.removeEventListener("pointercancel", finish); }; }, [finish]);

  const build = () => {
    const strokes = allCurrentStrokes, bbox = getSketchBounds(strokes, sceneSize.width, sceneSize.height);
    if (!bbox) return null;
    return { scene: "act3-forming-vision", round, colors: { primary: primaryColor, secondary: secondaryColor }, canvas: { width: sceneSize.width, height: sceneSize.height }, strokes: strokes.map((s) => ({ id: s.id, round: s.round, points: s.points.map((p) => ({ x: p.x, y: p.y, t: p.t, xNorm: p.x / sceneSize.width, yNorm: p.y / sceneSize.height })) })), bbox, sketchImageDataUrl: exportSketchImageData(strokes, sceneSize.width, sceneSize.height) };
  };

  const handleClear = () => { if (isRecognizing) return; setCurrentStrokes([]); setActiveStroke(null); };

  const handleConfirm = async () => {
    if (isRecognizing) return;
    const payload = build(); if (!payload) return;
    const strokesToConfirm = allCurrentStrokes;
    setDrawingEnabled(false); setIsRecognizing(true); setSideLines(["让我读一读。"]); setCenterLines([]);
    try {
      const result = onRecognizeSketch ? await onRecognizeSketch(payload) : await mockRecognizeSketch(payload);
      const item = { id: `${Date.now()}-${Math.random()}`, round, strokes: strokesToConfirm, bbox: payload.bbox, label: result.label || "桥", description: result.description || ["你画下了一个意象。", "它正在颜色里浮现。"], stylizedImageUrl: result.stylizedImageUrl || resolveObjectImage(result.label), overlay: result.overlay || { scale: 1.35, offsetX: 0, offsetY: 0 } };
      setConfirmedItems((p) => [...p, item]); setCurrentStrokes([]); setActiveStroke(null);
      const isLast = round >= maxRounds;
      setTimeout(() => {
        if (isLast) { setSideLines(["一个意象已经落下。", "它将被带往下一幕。"]); setCenterLines([]); setDrawingEnabled(false); setIsRecognizing(false); setTimeout(() => onComplete?.(), completeDelay); return; }
        setSideLines(["一个意象已经落下。", "你还想留下些什么？"]); setCenterLines(["继续画下另一个意象。", "画完后点击右下角确认。"]); setRound((p) => p + 1); setDrawingEnabled(true); setIsRecognizing(false);
      }, 2000);
    } catch (e) { console.error(e); setSideLines(["识别失败，请再试一次。"]); setCenterLines(["请重新绘制，或再次点击确认。"]); setDrawingEnabled(true); setIsRecognizing(false); }
  };

  return (
    <section ref={sceneRef} className="act3" onPointerDown={handleDown} onPointerMove={handleMove}>
      <img className="act3__bg" src={bgUrl} alt="" draggable="false" />
      <div className="act3__bar act3__bar--top" style={createMaskStyle(barTopUrl, primaryColor, 1)} />
      <div className="act3__bar act3__bar--topSoft" style={createMaskStyle(barTopSoftUrl, primaryColor, 0.3)} />
      <div className="act3__bar act3__bar--bottomSoft" style={createMaskStyle(barBottomSoftUrl, secondaryColor, 0.3)} />
      <div className="act3__bar act3__bar--bottom" style={createMaskStyle(barBottomUrl, secondaryColor, 1)} />
      <div className="act3__bar act3__bar--gradient" style={createMaskStyle(barGradientUrl, `linear-gradient(90deg, ${primaryColor} 0%, ${secondaryColor} 100%)`, 1)} />
      <img className="act3__title" src={titleUrl} alt="" draggable="false" />
      <div className="act3__sideCopy" key={sideLines.join("-")}>{sideLines.map((l, i) => <div key={`${l}-${i}`} className="act3__sideLine">{l}</div>)}</div>
      {centerLines.length > 0 && (<div className="act3__centerPrompt" key={centerLines.join("-")}>{centerLines.map((l, i) => <div key={`${l}-${i}`} className="act3__centerLine">{l}</div>)}</div>)}
      <canvas ref={canvasRef} className="act3__drawCanvas" />
      <div className="act3__resultsLayer">
        {confirmedItems.map((item) => {
          const vw = Math.max(item.bbox.normalized.width * 100 * (item.overlay?.scale || 1.35), 13);
          const vl = item.bbox.normalized.centerX * 100 + (item.overlay?.offsetX || 0);
          const vt = item.bbox.normalized.centerY * 100 + (item.overlay?.offsetY || 0);
          const side = vl > 60 ? "left" : "right";
          return (
            <div key={item.id} className="act3__resultItem">
              <img className="act3__resultVisual" src={item.stylizedImageUrl} alt={item.label} draggable="false" style={{ left: `${vl}%`, top: `${vt}%`, width: `${vw}%` }} />
              <div className={["act3__resultCaption", side === "left" ? "act3__resultCaption--left" : "act3__resultCaption--right"].join(" ")} style={{ left: `${vl}%`, top: `${vt}%` }}>{item.description.map((l, i) => <div key={`${item.id}-${i}`}>{l}</div>)}</div>
            </div>
          );
        })}
      </div>
      <div className="act3__toolbar">
        <div className="act3__roundBadge">第 {round} 次绘画</div>
        <button className="act3__toolBtn act3__toolBtn--ghost" type="button" onClick={handleClear} disabled={isRecognizing}>清除当前草图</button>
        <button className="act3__toolBtn act3__toolBtn--solid" type="button" onClick={handleConfirm} disabled={isRecognizing || allCurrentStrokes.length === 0}>{isRecognizing ? "识别中..." : "临时确认（握拳）"}</button>
      </div>
      <div className="act3__icons" aria-hidden="true">
        {floatingIcons.map((item) => (
          <img key={item.id} className="act3__icon" src={iconUrl} alt="" draggable="false" style={{ left: `${item.x}%`, top: `${item.y}%`, width: `${item.size}px`, height: `${item.size}px`, opacity: item.opacity, transform: `rotate(${item.rotate}deg)`, "--duration": `${item.duration}s`, "--delay": `${item.delay}s`, "--drift-x": `${item.driftX}px`, "--drift-y": `${item.driftY}px` }} />
        ))}
      </div>
    </section>
  );
}
