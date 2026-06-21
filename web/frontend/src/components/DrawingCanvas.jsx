import { useRef, useState } from "react";

export default function DrawingCanvas({
  mode,
  active,
  color,
  points,
  onPointsChange,
  onClear,
  onRecognize,
}) {
  const svgRef = useRef(null);
  const [drawing, setDrawing] = useState(false);

  const getPoint = (event) => {
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: Math.round(((event.clientX - rect.left) / rect.width) * 800),
      y: Math.round(((event.clientY - rect.top) / rect.height) * 460),
    };
  };

  const begin = (event) => {
    if (!active || mode !== "demo") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrawing(true);
    onPointsChange([...points, getPoint(event)]);
  };

  const move = (event) => {
    if (!drawing || !active || mode !== "demo") return;
    onPointsChange([...points, getPoint(event)]);
  };

  const end = () => setDrawing(false);
  const polyline = points.map((point) => `${point.x},${point.y}`).join(" ");

  return (
    <section className="panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="eyebrow">DRAWING CANVAS</p>
          <h2 className="mt-1 text-lg text-paper">绘制你心中的湖大</h2>
        </div>
        <div className="flex gap-2">
          <button className="button-secondary" onClick={onClear} disabled={!active}>清空重画</button>
          <button className="button-primary" onClick={onRecognize} disabled={!active || points.length === 0}>
            完成绘画
          </button>
        </div>
      </div>

      <div className="relative mt-4 overflow-hidden rounded-xl border border-white/10 bg-[#e7dec9]">
        <svg
          ref={svgRef}
          viewBox="0 0 800 460"
          className={`aspect-[800/460] w-full touch-none ${active && mode === "demo" ? "cursor-crosshair" : "cursor-default"}`}
          onPointerDown={begin}
          onPointerMove={move}
          onPointerUp={end}
          onPointerCancel={end}
          onPointerLeave={end}
          aria-label="鼠标绘画画布"
        >
          <defs>
            <pattern id="paper-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#7b705c" strokeOpacity="0.08" />
            </pattern>
          </defs>
          <rect width="800" height="460" fill="url(#paper-grid)" />
          <path d="M50 385 Q210 300 340 350 T750 320" fill="none" stroke="#35483a" strokeOpacity=".12" strokeWidth="36" />
          {points.length > 0 && (
            <polyline
              points={polyline}
              fill="none"
              stroke={color?.hex ?? "#333936"}
              strokeWidth="8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
        </svg>
        {points.length === 0 && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center text-center text-sm text-[#4c514c]/65">
            <p>{active ? "按住鼠标，在画布上留下轨迹" : "进入第二幕“筑景”后开始绘画"}</p>
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-white/45">
        <span>轨迹点：{points.length} · {mode === "demo" ? "鼠标模拟手势" : "等待 drawing_point 消息"}</span>
      </div>
    </section>
  );
}
