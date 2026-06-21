# SpiritModule / 《寻麓千年色》唤灵模块

`SpiritModule` 是第三幕“唤灵”的可移植 React 视觉模块。它只接收外部已经匹配好的人物数据，负责播放“寻找 → 剪影 → 台词 → 显影 → 身份揭示”的视觉流程。

模块不创建路由、不连接 WebSocket、不请求后端、不选择人物，也不会控制宿主项目的页面跳转。

## 复制与安装

把整个 `SpiritModule/` 文件夹复制到目标 React 项目的源码目录，例如：

```text
src/
├─ App.jsx
└─ SpiritModule/
```

不需要复制当前《寻麓千年色》项目的其他文件。

最低依赖：

```text
react >= 18.2
react-dom >= 18.2
```

模块没有 Tailwind、React Router、Redux、Zustand、Context 或其他第三方运行时依赖。宿主构建工具需要支持 JSX、CSS import 和 SVG import；Vite、Create React App、Webpack 等常见 React 工程默认支持。

## 导入

```jsx
import { SpiritPage } from "./SpiritModule";

function Example() {
  const character = {
    id: "zhang_shi",
    name: "张栻",
    title: "岳麓书院早期讲学者之一",
    image: "/characters/zhang_shi.webp",
    monologue: [
      "你选择了红。",
      "又画下讲堂。",
    ],
    spiritLine: [
      "刚才与你说话的，是他。",
      "但他留下的不只是名字，",
      "更是一种敢于发问的底色。",
    ],
  };

  return (
    <SpiritPage
      character={character}
      onComplete={(result) => {
        console.log("完成", result);
      }}
    />
  );
}
```

`SpiritPage.jsx` 会自动引入模块自己的 `SpiritPage.css`，宿主无需另外导入样式。

## Props

| Prop | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `character` | `object` | 必填 | 外部已经匹配好的人物 |
| `onComplete` | `(character) => void` | `undefined` | revealed 状态保持结束后调用 |
| `autoPlay` | `boolean` | `true` | 自动播放；为 `false` 时显示“开始唤灵”按钮 |
| `timings` | `object` | 见下文 | 覆盖各阶段时间 |
| `resolveCharacterImage` | `(character) => string` | `undefined` | 自定义人物图片地址 |
| `className` | `string` | `""` | 添加到模块最外层视口 |
| `style` | `object` | `undefined` | 添加到模块最外层视口 |

默认时间：

```js
{
  searchDuration: 1600,
  silhouetteDuration: 1200,
  sentenceDuration: 1200,
  revealDuration: 2200,
  revealedHoldDuration: 1500
}
```

自定义时间：

```jsx
<SpiritPage
  character={character}
  timings={{
    searchDuration: 1000,
    sentenceDuration: 900,
    revealDuration: 1800,
  }}
/>
```

没有传入的字段会继续使用默认值。

## character 数据格式

```js
{
  id: "zhang_shi",
  name: "张栻",
  title: "岳麓书院早期讲学者之一",
  image: "/characters/zhang_shi.webp",
  monologue: [
    "你选择了红。",
    "又画下讲堂。",
    "我知道那种颜色。"
  ],
  spiritLine: [
    "刚才与你说话的，是他。",
    "但他留下的不只是名字，",
    "更是一种敢于发问的底色。"
  ]
}
```

`monologue` 和 `spiritLine` 也可以传单个字符串。模块导出的 `normalizeCharacter` 可用于外部预处理：

```js
import { normalizeCharacter } from "./SpiritModule";

const safeCharacter = normalizeCharacter(apiCharacter);
```

## 人物图片

人物图片地址按以下顺序决定：

1. `character.image`
2. `resolveCharacterImage(character)` 的返回值
3. 模块内置的 `assets/default-silhouette.svg`

示例：

```jsx
<SpiritPage
  character={{ ...character, image: "" }}
  resolveCharacterImage={(item) =>
    `/characters/${item.id}.webp`
  }
/>
```

如果人物图片加载失败，组件会自动切换到内置默认剪影。整个流程始终使用同一个图片地址，只通过滤镜、透明度与缩放完成灰色模糊到清晰显影的变化。

## onComplete

模块不会自行跳转页面。流程完成后只通知宿主项目：

```jsx
<SpiritPage
  character={character}
  onComplete={(result) => {
    setCurrentStage("postcard");
    console.log("完成唤灵的人物", result);
  }}
/>
```

## 与 WebSocket 后端接入

WebSocket 必须由宿主项目管理。收到 `character_matched` 后，将人物对象传给模块即可：

```jsx
function PageWithSocket() {
  const [matchedCharacter, setMatchedCharacter] = React.useState(null);

  React.useEffect(() => {
    const socket = new WebSocket("wss://your-server.example/ws");

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "character_matched") {
        setMatchedCharacter(message.character);
      }
    };

    return () => socket.close();
  }, []);

  if (!matchedCharacter) return null;

  return <SpiritPage character={matchedCharacter} />;
}
```

上面的地址仅是宿主项目示例，`SpiritModule` 内部没有任何固定后端地址或 WebSocket 代码。

## 画布与排版坐标

模块内部使用固定 `1920 × 1080` 舞台，窗口变化时通过 `ResizeObserver` 对整个舞台等比例缩放，不会重新排列内部元素。

主要动态区域坐标：

| 区域 | left | top | width | height |
| --- | ---: | ---: | ---: | ---: |
| 人物 | 624 | 132 | 666 | 752 |
| 逐句台词 | 138 | 304 | 390 | 350 |
| 姓名与身份 | 158 | 678 | 365 | 205 |
| `????` | 840 | 855 | 210 | 70 |
| 右侧精神语句 | 1368 | 390 | 475 | 470 |
| 手动开始按钮 | 1600 | 914 | 230 | 58 |
| 状态标记 | 1680 | 1015 | 185 | 24 |

坐标统一维护在 `config/spiritLayout.js`。

## SVG 文件说明

- `assets/spirit-start-reference.svg`：原始开始状态参考稿，原样保存，不在运行时覆盖。
- `assets/spirit-end-reference.svg`：原始结束状态参考稿，原样保存，只用于核对排版。
- `assets/spirit-frame.svg`：正式运行使用的固定框架，不含固定人物、姓名、身份、台词、精神语句或 `????`。
- `assets/default-silhouette.svg`：人物图片缺失或加载失败时的回退图。

结束状态不会直接渲染 `spirit-end-reference.svg`，所以外部传入不同人物时，人物、姓名、身份和文字都会随数据变化。

## 开发预览

`demo/Demo.jsx` 和 `demo/demoCharacter.js` 只用于在宿主项目中临时预览：

```jsx
import Demo from "./SpiritModule/demo/Demo";
```

正式入口 `index.js` 和 `SpiritPage.jsx` 均不引用 demo 文件。
