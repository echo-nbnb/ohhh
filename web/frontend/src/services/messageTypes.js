// 前端内部事件类型
export const MESSAGE_TYPES = Object.freeze({
  COLOR_DETECTED: "color_detected",
  COLOR_DETECTION_ACTIVE: "color_detection_active",
  COLOR_DETECT_PROGRESS: "color_detect_progress",
  GESTURE_STATE: "gesture_state",
  DRAWING_START: "drawing_start",
  DRAWING_POINT: "drawing_point",
  OBJECT_RECOGNIZED: "object_recognized",
  CHARACTER_MATCHED: "character_matched",
  CHARACTERS_RECOMMENDED: "characters_recommended",
  NARRATIVE_GENERATED: "narrative_generated",
  POSTCARD_READY: "postcard_ready",
  SYSTEM_LOG: "system_log",
});

// 后端直通消息（不需要适配，原样传给前端）
export const BACKEND_PASSTHROUGH = Object.freeze({
  GESTURE_STATE: "gesture_state",
  HAND_TRACKING: "hand_tracking",
  HAND_APPEARED: "hand_appeared",
});

// 后端消息 → 适配后转发的类型集合
export const SUPPORTED_MESSAGE_TYPES = new Set([
  // 后端原始类型（适配器会转换）
  "color_extraction_start",
  "object_color_detected",
  "clothing_color_detected",
  "object_color_failed",
  "clothing_color_failed",
  "color_confirmed",
  "color_detection_active",
  "color_detect_progress",
  "drawing_start",
  "drawing_cancelled",
  "object_recognized",
  "object_confirmed",
  "objects_summary",
  "character_candidates",
  "character_search_start",
  "character_found",
  "character_performance",
  "character_revealed",
  "character_confirmed",
  "generation_result",
  "postcard_result",
  // 直通类型
  "connected",
  "gesture_state",
  "hand_tracking",
  "hand_appeared",
  "drawing_point",
]);
