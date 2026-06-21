import { colors } from "../data/colors";
import { recognizeObject } from "../data/objects";
import { matchCharacter } from "../data/characters";

export function mockDetectColor() {
  const color = colors[Math.floor(Math.random() * colors.length)];
  return {
    color,
    source: "Demo 物件扫描",
    confidence: 0.86,
  };
}

export function mockRecognizeObject(points, color) {
  return recognizeObject(points, color?.name ?? color ?? "");
}

export function mockMatchCharacter(color, objectResult) {
  return matchCharacter(color?.name ?? color ?? "", objectResult?.name ?? objectResult ?? "");
}

export function mockGenerateNarrative({ color, objectResult, matchedCharacter }) {
  return {
    title: `${color?.name ?? "千年色"} · ${objectResult?.name ?? "湖大"}之境`,
    summary: `以${color?.name ?? "用户底色"}为底，以${objectResult?.name ?? "绘制物象"}入景，形成属于用户的湖大千年色。`,
    paragraphs: [
      `系统从用户带来的物件中提取出一抹「${color?.name ?? "未知颜色"}」，它象征${color?.meaning ?? "这一刻的心绪"}。`,
      `用户在画布上留下轨迹，系统将其理解为「${objectResult?.name ?? "湖大物象"}」。`,
      `${matchedCharacter?.name ?? "一位湖湘先贤"}穿越文脉作出回应：${matchedCharacter?.spiritLine ?? "这次相遇留下了一道回声。"}`,
    ],
  };
}
