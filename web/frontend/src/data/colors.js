export const colors = [
  {
    id: "yuelu-green",
    name: "岳麓绿",
    hex: "#496b4a",
    meaning: "根脉、生长与绵延",
    voiceLine: "这是根脉的颜色。它来自树，也来自仍在生长的你。",
  },
  {
    id: "academy-red",
    name: "书院红",
    hex: "#8d3d36",
    meaning: "发问、理想与担当",
    voiceLine: "这是发问的颜色。红墙不只是建筑，它记住了许多人年轻时的理想。",
  },
  {
    id: "xiang-river-blue",
    name: "湘江蓝",
    hex: "#3f7082",
    meaning: "流动、远方与时间",
    voiceLine: "这是流动的颜色。它不停歇，也不回头。它把过去带向远方。",
  },
  {
    id: "westward-yellow",
    name: "西迁黄",
    hex: "#a9823e",
    meaning: "道路、坚韧与前行",
    voiceLine: "这是路的颜色。它不明亮，却坚定。它属于那些在风雨中仍选择前行的人。",
  },
  {
    id: "school-emblem-gold",
    name: "校徽金",
    hex: "#c3a45e",
    meaning: "理想、荣光与抵达",
    voiceLine: "这是理想的颜色。它不是炫耀，而是人在某一刻相信自己可以抵达更远处。",
  },
  {
    id: "ink",
    name: "墨色",
    hex: "#333936",
    meaning: "求索、沉思与答案",
    voiceLine: "这是求索的颜色。它很深，因为答案从来不浮在表面。",
  },
];

export const findColor = (name) =>
  colors.find((item) => item.name === name) ?? colors[0];
