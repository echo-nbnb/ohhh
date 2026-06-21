import spiritFrame from "../assets/spirit-frame.svg";
import { SPIRIT_STAGE_SIZE } from "../config/spiritLayout";
import { useStageScale } from "../hooks/useStageScale";

export default function SpiritCanvas({ children, className = "", style }) {
  const { viewportRef, scale, scaledWidth, scaledHeight } = useStageScale();

  return (
    <div
      ref={viewportRef}
      className={`spirit-module-viewport ${className}`.trim()}
      style={style}
    >
      <div
        className="spirit-module-stage-shell"
        style={{ width: scaledWidth, height: scaledHeight }}
      >
        <div
          className="spirit-module-stage"
          style={{
            width: SPIRIT_STAGE_SIZE.width,
            height: SPIRIT_STAGE_SIZE.height,
            transform: `scale(${scale})`,
          }}
        >
          <img
            className="spirit-module-frame"
            src={spiritFrame}
            alt=""
            aria-hidden="true"
          />
          <div className="spirit-module-interaction-layer">{children}</div>
        </div>
      </div>
    </div>
  );
}
