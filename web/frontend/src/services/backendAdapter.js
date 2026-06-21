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
      };

    case "color_confirmed":
      return {
        type: MESSAGE_TYPES.COLOR_DETECTED,
        colorName: asStr(payload.color),
        source: asStr(payload.source, "confirmed"),
        confidence: 1.0,
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
    case "character_revealed":
    case "character_confirmed":
      return {
        type: MESSAGE_TYPES.SYSTEM_LOG,
        level: "info",
        message: asStr(payload.message),
      };

    // ---- Hand tracking → drawing_point (食指指尖 landmarks[8]) ----
    case "hand_tracking": {
      const ft = payload.fingertips;
      if (Array.isArray(ft) && ft.length >= 2) {
        const tip = ft[1]; // 食指
        if (Array.isArray(tip) && tip.length >= 2) return { type: MESSAGE_TYPES.DRAWING_POINT, x: tip[0], y: tip[1] };
        if (tip && typeof tip === "object") return { type: MESSAGE_TYPES.DRAWING_POINT, x: tip.x ?? tip[0] ?? 0, y: tip.y ?? tip[1] ?? 0 };
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

    // ---- Character performance → character_matched (如没 candidates 的兜底) ----
    case "character_performance":
      return {
        type: MESSAGE_TYPES.CHARACTER_MATCHED,
        character: {
          name: "",
          title: "",
          reason: "",
          monologue: asArr(payload.paragraphs),
          spiritLine: "",
        },
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
