export const colors = [
  // 第一组 · 亮色系
  { id: "vermillion",   name: "朱红", hex: "#DD0101", meaning: "热烈、赤诚、初心" },
  { id: "lantern",      name: "灯橙", hex: "#FE6600", meaning: "温暖、灯火、守望" },
  { id: "pear-yellow",  name: "梨黄", hex: "#F0E440", meaning: "明净、收获、沉淀" },
  { id: "leaf-green",   name: "叶绿", hex: "#6CFF2D", meaning: "生长、生机、蔓延" },
  { id: "porcelain",    name: "瓷青", hex: "#25FFF4", meaning: "清透、澄澈、明镜" },
  { id: "sea-blue",     name: "海蓝", hex: "#1F6AFF", meaning: "深邃、探索、求知" },
  { id: "smoke-purple", name: "烟紫", hex: "#7B00FF", meaning: "神秘、思辨、超越" },
  // 第二组 · 浓色系
  { id: "maple-red",    name: "枫红", hex: "#C40000", meaning: "深沉、厚重、积淀" },
  { id: "warm-orange",  name: "暖橙", hex: "#FFB432", meaning: "醇厚、温情、守候" },
  { id: "vine-yellow",  name: "藤黄", hex: "#F6D900", meaning: "古雅、文脉、传承" },
  { id: "jade-green",   name: "玉绿", hex: "#00EB0C", meaning: "温润、清雅、含蓄" },
  { id: "stone-cyan",   name: "石青", hex: "#3BFDBB", meaning: "沉稳、坚定、磐石" },
  { id: "pure-blue",    name: "澄蓝", hex: "#213CEB", meaning: "纯粹、追问、求真" },
  { id: "shadow-purple",name: "影紫", hex: "#5100BB", meaning: "幽深、内省、潜思" },
  // 第三组 · 柔色系
  { id: "peach-pink",   name: "桃红", hex: "#FF4141", meaning: "柔和、青春、绽放" },
  { id: "dusk-orange",  name: "夕橙", hex: "#E76204", meaning: "黄昏、余韵、回望" },
  { id: "osmanthus",    name: "桂黄", hex: "#FFEE00", meaning: "金桂、荣光、远方" },
  { id: "tea-green",    name: "茶绿", hex: "#38FF7E", meaning: "淡雅、从容、澄怀" },
  { id: "lake-cyan",    name: "湖青", hex: "#3EF9FF", meaning: "涟漪、流动、包容" },
  { id: "vast-blue",    name: "沧蓝", hex: "#4472FF", meaning: "苍茫、广阔、志向" },
  { id: "elegant-purple",name: "黛紫", hex: "#9E00E7", meaning: "典雅、深邃、余韵" },
];

export const findColor = (name) =>
  colors.find((item) => item.name === name) ?? colors[0];
