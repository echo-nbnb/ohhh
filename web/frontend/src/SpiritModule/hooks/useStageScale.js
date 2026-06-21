import { useEffect, useRef, useState } from "react";
import { SPIRIT_STAGE_SIZE } from "../config/spiritLayout";

export function useStageScale() {
  const viewportRef = useRef(null);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return undefined;

    const update = () => {
      const { width, height } = viewport.getBoundingClientRect();
      if (!width || !height) return;
      setScale(
        Math.min(
          width / SPIRIT_STAGE_SIZE.width,
          height / SPIRIT_STAGE_SIZE.height,
        ),
      );
    };

    update();

    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(update);
      observer.observe(viewport);
      return () => observer.disconnect();
    }

    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  return {
    viewportRef,
    scale,
    scaledWidth: SPIRIT_STAGE_SIZE.width * scale,
    scaledHeight: SPIRIT_STAGE_SIZE.height * scale,
  };
}
