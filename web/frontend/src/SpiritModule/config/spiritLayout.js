export const SPIRIT_STAGE_SIZE = Object.freeze({
  width: 1920,
  height: 1080,
});

export const SPIRIT_LAYOUT = Object.freeze({
  portrait: { left: 624, top: 132, width: 666, height: 752 },
  monologue: { left: 138, top: 304, width: 390, height: 350 },
  identity: { left: 158, top: 678, width: 365, height: 205 },
  unknownName: { left: 840, top: 855, width: 210, height: 70 },
  message: { left: 1368, top: 390, width: 475, height: 470 },
  startButton: { left: 1600, top: 914, width: 230, height: 58 },
  status: { left: 1680, top: 1015, width: 185, height: 24 },
});

export const DEFAULT_TIMINGS = Object.freeze({
  searchDuration: 1600,
  silhouetteDuration: 1200,
  sentenceDuration: 1200,
  revealDuration: 2200,
  revealedHoldDuration: 1500,
});

export function regionStyle(region) {
  return {
    position: "absolute",
    left: region.left,
    top: region.top,
    width: region.width,
    height: region.height,
  };
}
