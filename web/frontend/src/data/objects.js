export const objects = [
  { name: "古树", family: "vertical", narrative: "你画下了一株古树。枝叶向上，根系向下，把今天与千年前的山林连在一起。" },
  { name: "桥梁", family: "horizontal", narrative: "你画下了一座桥。桥连接两岸，也连接出发与归来。" },
  { name: "流水", family: "horizontal", narrative: "你留下了一道流水。它经过书院，也把未说完的话带向远方。" },
  { name: "碑刻", family: "vertical", narrative: "你画下了一方碑刻。石头沉默，却替时间保存了人的声音。" },
  { name: "岳麓山", family: "closed", narrative: "线条回到起点，像岳麓山收拢群峰，也收拢这一刻的心绪。" },
  { name: "书卷", family: "vertical", narrative: "你画下了一卷书。纸页展开，像一场跨越千年的问答。" },
  { name: "林荫道", family: "horizontal", narrative: "你画下了一条林荫道。道路伸向深处，脚步也因此有了方向。" },
  { name: "湖面", family: "closed", narrative: "线条围出一片湖面。水光收下倒影，也收下你的颜色。" },
];

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export function recognizeObject(points, colorName = "") {
  if (points.length < 5) {
    return { name: "书卷", score: 0.58, reason: "轨迹较短，像一页刚刚展开的书卷。", narrative: objects[5].narrative };
  }

  const first = points[0];
  const last = points[points.length - 1];
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const closed = distance(first, last) < Math.max(36, Math.min(width, height) * 0.35);

  let family = closed ? "closed" : width > height * 1.25 ? "horizontal" : "vertical";
  const candidates = objects.filter((item) => item.family === family);

  // 多样性: 轨迹特征哈希 + 颜色种子 + 时间噪声, 避免总是同一结果
  const trajHash = points.reduce((h, p, i) => h + Math.round(p.x * 7 + p.y * 13) * (i + 1), 0);
  const colorSeed = [...colorName].reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const timeNoise = Math.floor(Date.now() / 3000) % 97; // 3秒窗口变化
  const idx = (trajHash + colorSeed + timeNoise) % candidates.length;
  const selected = candidates[idx];

  return {
    ...selected,
    score: Math.min(0.96, 0.7 + points.length / 600),
    reason: closed
      ? "轨迹首尾接近，形成了被山水环抱的意象。"
      : family === "horizontal"
        ? "轨迹横向延展，呈现道路、水流或桥梁的方向感。"
        : "轨迹纵向生长，接近古树、碑刻或书卷的意象。",
  };
}
