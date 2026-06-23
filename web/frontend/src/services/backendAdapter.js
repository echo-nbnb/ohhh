import { MESSAGE_TYPES, SUPPORTED_MESSAGE_TYPES } from "./messageTypes";

const asStr = (v, fb = "") => (typeof v === "string" ? v : fb);
const asNum = (v, fb = 0) => { const n = Number(v); return Number.isFinite(n) ? n : fb; };
const asArr = (v) => Array.isArray(v) ? v.filter(x => typeof x === "string") : [];

// ================================================================
// 后端原始消息 → 前端内部事件
// ================================================================

export function normalizeBackendMessage(payload) {
  if (!payload || typeof payload !== "object" || !SUPPORTED_MESSAGE_TYPES.has(payload.type)) {
    return null;
  }

  switch (payload.type) {

    // ---- Color: object_color_detected / clothing / confirmed → color_detected ----
    case "object_color_detected":
    case "clothing_color_detected":
      return {
        type: MESSAGE_TYPES.COLOR_DETECTED,
        colorName: asStr(payload.color),
        source: asStr(payload.source, "object"),
        confidence: asNum(payload.confidence),
        round: asNum(payload.round, 1),
      };

    case "color_confirmed":
      return {
        type: MESSAGE_TYPES.COLOR_DETECTED,
        colorName: asStr(payload.color),
        source: asStr(payload.source, "confirmed"),
        confidence: 1.0,
      };

    // ---- Color detection start / progress → color_detection_active / color_detect_progress ----
    case "color_detection_active":
      return {
        type: MESSAGE_TYPES.COLOR_DETECTION_ACTIVE,
      };

    case "color_detect_progress":
      return {
        type: MESSAGE_TYPES.COLOR_DETECT_PROGRESS,
        stableColor: asStr(payload.stable_color),
        elapsed: asNum(payload.elapsed),
        confirmSeconds: asNum(payload.confirm_seconds, 3),
      };

    // ---- Color extraction / fallback → system_log ----
    case "color_extraction_start":
    case "object_color_failed":
    case "clothing_color_failed":
    case "drawing_start":
    case "drawing_cancelled":
    case "object_confirmed":
    case "objects_summary":
    case "character_search_start":
    case "character_found":
    case "character_confirmed":
      return {
        type: MESSAGE_TYPES.SYSTEM_LOG,
        level: "info",
        message: asStr(payload.message),
      };

    // ---- Character revealed → 更新人物到前端 ----
    case "character_revealed":
      return {
        type: MESSAGE_TYPES.CHARACTER_MATCHED,
        character: {
          name: asStr(payload.name, "回应者"),
          title: asStr(payload.title),
          reason: asStr(payload.message),
          portrait: asStr(payload.portrait, ""),
          monologue: asArr(payload.monologue ?? payload.paragraphs ?? payload.performance),
          spiritLine: asStr(payload.spiritLine ?? payload.spirit_line ?? payload.summary, ""),
        },
      };

    // ---- drawing_point (真实摄像头直接发送) ----
    case "drawing_point":
      return {
        type: MESSAGE_TYPES.DRAWING_POINT,
        x: asNum(payload.x),
        y: asNum(payload.y),
      };

    // ---- Hand tracking → drawing_point (备选: 从手部通道提取食指指尖) ----
    case "hand_tracking": {
      // 优先取 fingertips
      const ft = payload.fingertips;
      if (Array.isArray(ft) && ft.length >= 4) {
        // 平面数组 [thumb_x, thumb_y, index_x, index_y, ...]
        if (typeof ft[0] === "number") return { type: MESSAGE_TYPES.DRAWING_POINT, x: ft[2], y: ft[3] };
        // 嵌套数组 [[x,y], [x,y], ...]
        const tip = ft[1]; // 食指 (index 1)
        if (Array.isArray(tip) && tip.length >= 2) return { type: MESSAGE_TYPES.DRAWING_POINT, x: tip[0], y: tip[1] };
        if (tip && typeof tip === "object") return { type: MESSAGE_TYPES.DRAWING_POINT, x: tip.x ?? tip[0] ?? 0, y: tip.y ?? tip[1] ?? 0 };
      }
      // Fallback: 从 landmarks (平铺数组 [x0,y0,...,x20,y20]) 取食指指尖 index=8
      const lm = payload.landmarks;
      if (Array.isArray(lm) && lm.length >= 18) {
        // 坐标已归一化 0-1，直接透传
        return { type: MESSAGE_TYPES.DRAWING_POINT, x: lm[16], y: lm[17] };
      }
      return null;
    }

    // ---- Object recognized → object_recognized ----
    case "object_recognized": {
      const obj = payload.object ?? {};
      const name = asStr(obj.name ?? payload.name, "未知物象");
      return {
        type: MESSAGE_TYPES.OBJECT_RECOGNIZED,
        name,
        score: asNum(obj.score ?? payload.score),
        reason: asStr(payload.narration, `你画下了${name}。`),
        narrative: asStr(payload.narration, `你画下了${name}。`),
      };
    }

    // ---- Character candidates → character_matched (取首位，打包) ----
    case "character_candidates": {
      const cs = payload.candidates ?? [];
      if (cs.length === 0) return { type: MESSAGE_TYPES.CHARACTER_MATCHED, character: null };
      const c = cs[0];
      return {
        type: MESSAGE_TYPES.CHARACTER_MATCHED,
        character: {
          name: asStr(c.name, "回应者"),
          title: asStr(c.title),
          reason: asStr(c.reason),
          monologue: asArr(c.monologue ?? c.performance),
          spiritLine: asStr(c.spiritLine ?? c.spirit_line, ""),
        },
      };
    }

    // ---- Character performance → system_log (不覆盖已设置的人物) ----
    case "character_performance":
      return {
        type: MESSAGE_TYPES.SYSTEM_LOG,
        level: "info",
        message: asArr(payload.paragraphs).join(" "),
      };

    // ---- Generation → narrative_generated ----
    case "generation_result":
      return {
        type: MESSAGE_TYPES.NARRATIVE_GENERATED,
        title: asStr(payload.title, "你寻到的千年色"),
        summary: asStr(payload.narrative ?? payload.summary),
        paragraphs: asArr(payload.paragraphs),
      };

    // ---- Postcard / QR → postcard_ready ----
    case "postcard_result":
      return {
        type: MESSAGE_TYPES.POSTCARD_READY,
        imageUrl: asStr(payload.image_url),
        qrBase64: asStr(payload.qr_base64),
        uniqueId: asStr(payload.unique_id),
      };

    // ---- Backend connected → 重置前端到第一幕 ----
    case "connected":
      return {
        type: MESSAGE_TYPES.SYSTEM_LOG,
        level: "info",
        message: "后端已就绪，等待择色…",
      };

    // ---- Gesture state 直通 ----
    case "gesture_state":
      return {
        type: MESSAGE_TYPES.GESTURE_STATE,
        mode: asStr(payload.mode),
        gesture: asStr(payload.gesture ?? payload.mode),
      };

    // ---- Hand appeared → log ----
    case "hand_appeared":
      return { type: MESSAGE_TYPES.SYSTEM_LOG, level: "info", message: "手部进入检测区" };

    default:
      return null;
  }
}
