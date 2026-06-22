export const characters = [
  {
    name: "张栻",
    title: "岳麓书院早期讲学者之一",
    reason: "书院红与讲堂共同指向发问、讲学与书院精神。",
    monologue: ["你选择了一种不肯沉默的颜色。", "又画下通往答案的形状。", "我知道那不是热闹的勇气。", "而是站在众人面前，仍愿意发问。"],
    spiritLine: "他留下的不只是名字，而是一种敢于发问的颜色。",
    colors: ["书院红", "墨色"],
    objects: ["书卷", "碑刻", "桥梁"],
  },
  {
    name: "朱熹",
    title: "湖湘会讲的重要思想家",
    reason: "深色与书卷让求索从表面进入事理深处。",
    monologue: ["你把颜色沉了下来。", "像墨落在纸上，也像问题落在心里。", "不要急着找到答案。", "先看清它为何值得追问。"],
    spiritLine: "求索不是抵达一个结论，而是让心不断接近事理。",
    colors: ["墨色", "书院红"],
    objects: ["书卷", "碑刻"],
  },
  {
    name: "王夫之",
    title: "明清之际思想家",
    reason: "山林与墨色对应坚守、思辨和在动荡中的自持。",
    monologue: ["山色很深。", "你画下的线条却没有停。", "世事会变化，人的判断也要经受风雨。", "守住的不是旧答案，而是求真的心。"],
    spiritLine: "在变化中保持清醒，也是一种坚韧的颜色。",
    colors: ["岳麓绿", "墨色"],
    objects: ["岳麓山", "古树"],
  },
  {
    name: "胡宏",
    title: "湖湘学派重要奠基者",
    reason: "岳麓绿与古树指向根脉、性理和持续生长。",
    monologue: ["我看见你画下生长。", "根在看不见的地方延伸。", "枝叶才会在光里展开。", "人心也有这样的根。"],
    spiritLine: "真正的生长，先发生在看不见的根脉里。",
    colors: ["岳麓绿"],
    objects: ["古树", "林荫道", "岳麓山"],
  },
  {
    name: "毛泽东",
    title: "曾在岳麓山求学问道的青年",
    reason: "湘江、道路与青春理想共同指向实践和远方。",
    monologue: ["你画的水向前流。", "它不会替人选择方向。", "但年轻的心一旦看见远方，", "便会在行走中回答自己的问题。"],
    spiritLine: "理想不是悬在高处的光，而是决定迈出的下一步。",
    colors: ["湘江蓝", "校徽金"],
    objects: ["流水", "桥梁", "林荫道"],
  },
  {
    name: "魏源",
    title: "近代经世思想家",
    reason: "流动的蓝与远行之路，指向睁眼看世界的求变精神。",
    monologue: ["水把消息带来，也把人送往远处。", "你画出的边界不是终点。", "看见更大的世界，", "才能重新理解脚下的土地。"],
    spiritLine: "向外看，是为了更清醒地回答从哪里来、往哪里去。",
    colors: ["湘江蓝", "西迁黄"],
    objects: ["流水", "桥梁", "湖面"],
  },
  {
    name: "李达",
    title: "湖南大学首任校长之一",
    reason: "校徽金与书卷相遇，连接现代大学的理想与实践。",
    monologue: ["你把一页书变成了一条路。", "大学之大，不只在楼宇。", "它应当让知识进入时代，", "也让每个人找到自己的责任。"],
    spiritLine: "大学的光，来自知识与时代彼此照见。",
    colors: ["校徽金", "书院红"],
    objects: ["书卷", "桥梁", "碑刻"],
  },
  {
    name: "杨昌济",
    title: "近代教育家、伦理学者",
    reason: "理想之金与林荫道路，指向教育、修身和青年成长。",
    monologue: ["你选择了一束克制的光。", "它照见的不是荣耀本身。", "而是一个人愿意怎样成为自己，", "又愿意为世界承担什么。"],
    spiritLine: "教育让理想有了尺度，也让行动有了方向。",
    colors: ["校徽金", "西迁黄"],
    objects: ["林荫道", "书卷", "古树"],
  },
];

export function matchCharacter(colorName, objectName) {
  const scored = characters.map((character) => {
    // 精确匹配 2 分，颜色名称包含匹配 1 分（如"湘江蓝"包含"蓝"字，匹配"海蓝"或"澄蓝"）
    const colorScore = character.colors.includes(colorName) ? 2
      : character.colors.some((c) => colorName.includes(c.slice(-1)) || c.includes(colorName.slice(-1))) ? 1
      : 0;
    const objectScore = character.objects.includes(objectName) ? 2
      : 0;
    return { character, score: colorScore + objectScore };
  });

  // 按分数降序排列
  scored.sort((a, b) => b.score - a.score);

  // 多样性: 同分角色中随机选取，而非总是第一个
  const topScore = scored[0].score;
  const topCandidates = scored.filter((item) => item.score === topScore);
  const timeNoise = Math.floor(Date.now() / 5000) % 31; // 5秒窗口变化
  const pick = topCandidates[(timeNoise + topCandidates.length) % topCandidates.length];

  return pick.character;
}
