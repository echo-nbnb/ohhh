import { useCallback, useEffect, useRef, useState } from "react";
import { findColor } from "./data/colors";
import { useWebSocket } from "./hooks/useWebSocket";
import { normalizeBackendMessage } from "./services/backendAdapter";
import {
  mockDetectColor,
  mockGenerateNarrative,
  mockMatchCharacter,
  mockRecognizeObject,
} from "./services/mockEngine";
import { MESSAGE_TYPES } from "./services/messageTypes";

import Act0 from "./pages/Act0/Act0";
import Act1Entry from "./pages/Act1Entry/Act1Entry";
import Act2ColorSeeking from "./pages/Act2ColorSeeking/Act2ColorSeeking";
import Act3FormingVision from "./pages/Act3FormingVision/Act3FormingVision";
import Act4SpiritCalling from "./pages/Act4SpiritCalling/Act4SpiritCalling";
import Act5Postcard from "./pages/Act5Postcard/Act5Postcard";

const WS_URL = (() => {
  const { protocol, host } = window.location;
  return `${protocol === "https:" ? "wss" : "ws"}://${host}/ws`;
})();

const STAGES = {
  INTRO: "intro", TRANSITION: "transition", COLOR: "color",
  DRAW: "draw", SPIRIT: "spirit", POSTCARD: "postcard",
};

// ── Helpers ──

function buildAct2CopyByStep(colors) {
  const c0 = colors[0];
  const c1 = colors[1];
  if (!c0) {
    return { 1: ["请将随身之物靠近光中。", "让它替你说话。"], 2: ["让我看看……", "这件东西里", "藏着怎样的颜色。"], 3: [], 4: [] };
  }
  if (!c1) {
    return {
      1: ["请将随身之物靠近光中。", "让它替你说话。"], 2: ["让我看看……", "这件东西里", "藏着怎样的颜色。"],
      3: [`一缕${c0.name}浮出光面，`, `像旧纸上醒来的日色。`],
      4: [`请将另一件随身之物靠近光中。`, `再寻一色，让它们相遇。`],
    };
  }
  return {
    1: ["请将随身之物靠近光中。", "让它替你说话。"], 2: ["让我看看……", "这件东西里", "藏着怎样的颜色。"],
    3: [`一缕${c0.name}浮出光面，`, `像旧纸上醒来的日色。`],
    4: [`${c0.name}与${c1.name}相遇，`, `像山门灯火照见夜色。`],
  };
}

function buildAct4Payload(colors, objectResult) {
  const c0 = colors[0];
  const c1 = colors[1] || c0;
  const objs = objectResult?.name || "桥";
  return {
    primaryColor: c0?.hex || "#F2E700",
    secondaryColor: c1?.hex || "#355BFF",
    firstColorName: c0?.name || "黄",
    secondColorName: c1?.name || "蓝",
    firstImageryName: objs,
    secondImageryName: objs,
  };
}

function buildAct5Data(colors, imageryItems, matchedCharacter, narrative, qrBase64, imageUrl) {
  const c0 = colors[0];
  const c1 = colors[1] || c0;
  console.log("[buildAct5Data] colors:", colors.map(c => c?.hex), "c0:", c0?.hex, "c1:", c1?.hex);
  const now = new Date();
  const dateStr = `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, "0")}.${String(now.getDate()).padStart(2, "0")}\n${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  return {
    colors: { primary: c0?.hex || "#F2E700", secondary: c1?.hex || "#355BFF", primaryName: c0?.name || "明黄", secondaryName: c1?.name || "澄蓝" },
    selectedPlaces: ["岳麓山", "湘江水", "书院檐角"],
    title: { cn: narrative?.title || `${c0?.name || "千年色"}之境`, en: "THE MILLENNIUM COLOR OF YOURS" },
    traceText: `[${c0?.name || "?"}｜${c1?.name || "?"}] → [${matchedCharacter?.name || "?"}]`,
    imageryItems: imageryItems.length > 0 ? imageryItems : [{ id: "obj", name: "意象", imageUrl: "", className: "act5__imagery--bridge" }, { id: "tree", name: "树", imageUrl: "", className: "act5__imagery--tree" }],
    objectText: imageryItems[0]?.description || ["这个意象已经落下。"],
    aiWriting: (() => {
      const names = imageryItems.map(it => it.name).filter(Boolean);
      if (names.length >= 2 && narrative?.paragraphs) {
        // Replace backend text's object references with deduplicated names
        return narrative.paragraphs.map(p => {
          let txt = p;
          // Replace "X、Y" pattern with deduplicated names
          const joined = names.join("、");
          txt = txt.replace(/[^\s、]+、[^\s]+/g, joined);
          return txt;
        });
      }
      return narrative?.paragraphs || ["你的千年色正在成形……"];
    })(),
    person: { name: matchedCharacter?.name || "回应者", portraitUrl: matchedCharacter?.portrait ? `/src/assets/act5/people/${matchedCharacter.portrait}.png` : "" },
    mainTitleImageUrl: "", downloadQrUrl: qrBase64 || "",
    createdAtText: dateStr,
  };
}

export default function App() {
  const mode = "live";
  const [currentStage, setCurrentStage] = useState(STAGES.INTRO);
  const [colors, setColors] = useState([]);          // [color1, color2?]
  const [colorSource, setColorSource] = useState(null);
  const [gesture, setGesture] = useState(null);
  const [points, setPoints] = useState([]);
  const [objectResult, setObjectResult] = useState(null);
  const [imageryItems, setImageryItems] = useState([]); // [{name, description, bbox}]
  const [matchedCharacter, setMatchedCharacter] = useState(null);
  const [spiritStatus, setSpiritStatus] = useState("idle");
  const [narrative, setNarrative] = useState(null);
  const [isAutoAdvancing, setIsAutoAdvancing] = useState(false);
  const [colorStep, setColorStep] = useState(1);
  const [showAct1Transition, setShowAct1Transition] = useState(true);
  const [waitingForStamp, setWaitingForStamp] = useState(false); // Act4 等握拳盖章
  const [postcardQr, setPostcardQr] = useState("");       // QR base64 from backend
  const [postcardImageUrl, setPostcardImageUrl] = useState(""); // postcard image URL
  const [act3Overlay, setAct3Overlay] = useState(null);    // Act3 canvas screenshot for overlay
  const postcardEnterTimeRef = useRef(0);    // when Act5 was entered

  // ── Act2 颜色检测状态 ──
  const [isDetecting, setIsDetecting] = useState(false);
  const [colorRound, setColorRound] = useState(1);       // 当前第几色
  const [firstColor, setFirstColor] = useState(null);     // {hex, name}
  const [secondColor, setSecondColor] = useState(null);   // {hex, name}
  const [stableColorName, setStableColorName] = useState(null);
  const [stableSeconds, setStableSeconds] = useState(0);
  const confirmSeconds = 3;

  const timersRef = useRef([]);
  const latestRef = useRef({ mode, colors, objectResult, matchedCharacter, narrative, imageryItems });
  const colorsRef = useRef(colors);
  const stageRef = useRef(currentStage);
  const goToTransitionRef = useRef(null);
  const showAct1TransitionRef = useRef(showAct1Transition);
  const fistTriggeredRef = useRef(false);
  const colorLockedRef = useRef(false);
  const goToSpiritRef = useRef(null);
  const liveObjectCountRef = useRef(0);      // track objects confirmed in live mode
  const usedObjectNamesRef = useRef([]);     // track used object names for dedup
  const fallbackTimerRef = useRef(null);     // cancelable fallback color timer
  const drawThrottleRef = useRef(null);      // throttle drawing_point updates
  const drawPositionsRef = useRef([]);        // accumulate points for position calc
  const resetCooldownUntil = useRef(0);      // ignore gestures/colors until timestamp
  const remoteDrawRef = useRef(null);         // direct canvas draw callback (bypass React)
  const colorConfirmedRef = useRef(false);     // 后端确认颜色后才允许进入 Act3

  // ── 键盘控制 ──
  // R = 检测颜色（Act2），Space = 推进下一幕
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      const s = stageRef.current;
      // R 键: 触发后端真实颜色检测，检测两次完成择色
      if (e.code === "KeyR" && s === STAGES.COLOR) {
        e.preventDefault();
        wsSend({type: "trigger_color_detect"});
        addLog("触发颜色检测...");
        return;
      }
      // Space: 推进
      if (e.code !== "Space") return;
      e.preventDefault();
      if (s === STAGES.INTRO || s === STAGES.TRANSITION) goToColor();
      else if (s === STAGES.COLOR) { colorConfirmedRef.current = true; goToDraw(); }
      else if (s === STAGES.DRAW) goToSpirit();
      else if (s === STAGES.SPIRIT) goToPostcard();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => { colorsRef.current = colors; }, [colors]);

  useEffect(() => { latestRef.current = { mode, colors, objectResult, matchedCharacter, narrative, imageryItems }; }, [mode, colors, objectResult, matchedCharacter, narrative, imageryItems]);
  useEffect(() => { stageRef.current = currentStage; }, [currentStage]);
  useEffect(() => { showAct1TransitionRef.current = showAct1Transition; }, [showAct1Transition]);

  const clearTimers = useCallback(() => { timersRef.current.forEach(clearTimeout); timersRef.current = []; }, []);
  const schedule = useCallback((cb, delay) => { const t = setTimeout(() => { timersRef.current = timersRef.current.filter(i => i !== t); cb(); }, delay); timersRef.current.push(t); return t; }, []);
  useEffect(() => clearTimers, [clearTimers]);

  const addLog = useCallback((msg) => { console.log(`[${new Date().toLocaleTimeString("zh-CN", { hour12: false })}]`, msg); }, []);

  // ── Color detection (supports 2 colors) ──
  const applyColorResult = useCallback((detectedColor, source, confidence = null) => {
    if (mode === "live") {
      const nextColors = (prev) => {
        if (prev.length >= 2) return prev;
        if (!prev.some(c => c.name === detectedColor.name)) return [...prev, detectedColor];
        // Same color detected — auto-pick a different one as second
        const allColors = ["岳麓绿","书院红","湘江蓝","西迁黄","校徽金","墨色"];
        const unused = allColors.filter(n => n !== detectedColor.name && !prev.some(c => c.name === n));
        const fbName = unused[Math.floor(Math.random() * unused.length)];
        const fbColor = findColor(fbName);
        addLog(`第二色与第一色相同，自动备选：${fbName}`);
        return [...prev, fbColor];
      };
      setColors(nextColors);
      setColorSource(confidence === null ? source : `${source} · ${Math.round(confidence * 100)}%`);
      const newLen = Math.min(colorsRef.current.length + 1, 2);
      addLog(`择色${newLen}：读取为${detectedColor.name}`);
      if (newLen === 1) {
        schedule(() => setColorStep(2), 3000);
        schedule(() => setColorStep(3), 8000);
        schedule(() => setColorStep(4), 12000);
        // Cancelable fallback: pick a DIFFERENT color if second not detected in time
        if (fallbackTimerRef.current) clearTimeout(fallbackTimerRef.current);
        fallbackTimerRef.current = setTimeout(() => {
          if (colorsRef.current.length < 2) {
            const allColors = ["岳麓绿","书院红","湘江蓝","西迁黄","校徽金","墨色"];
            const unused = allColors.filter(n => n !== detectedColor.name);
            const fbName = unused[Math.floor(Math.random() * unused.length)];
            const fbColor = findColor(fbName);
            setColors(prev => prev.length < 2 ? [...prev, fbColor] : prev);
            setColorStep(4);
            addLog(`第二色超时，备选：${fbName}`);
          }
        }, 20000);
      } else if (newLen === 2) {
        // Second real color arrived — cancel fallback and jump to step 4
        if (fallbackTimerRef.current) { clearTimeout(fallbackTimerRef.current); fallbackTimerRef.current = null; }
        setColorStep(4);
      }
      return;
    }
    // Demo mode
    clearTimers();
    setCurrentStage(STAGES.COLOR);
    setColors([detectedColor]);
    setColorSource(confidence === null ? source : `${source} · ${Math.round(confidence * 100)}%`);
    setPoints([]); setObjectResult(null); setImageryItems([]);
    setMatchedCharacter(null); setSpiritStatus("idle"); setNarrative(null);
    setIsAutoAdvancing(true);
    addLog(`择色完成：读取为${detectedColor.name}`);
    setColorStep(2);
    schedule(() => setColorStep(3), 2500);
    schedule(() => setColorStep(4), 5000);
  }, [addLog, clearTimers, schedule, mode, colors.length]);

  // ── Stage transitions ──
  const goToColor = useCallback(() => {
    setCurrentStage(STAGES.COLOR); setColorStep(1);
    addLog("入境完成，进入寻色");
    if (mode === "demo") {
      const r = mockDetectColor();
      schedule(() => applyColorResult(r.color, r.source, r.confidence), 3000);
    }
    // Live mode: always drive step progression (even with 0 colors)
    if (mode === "live") {
      if (colors.length === 0) {
        schedule(() => setColorStep(2), 3000);
        schedule(() => setColorStep(3), 8000);
        schedule(() => setColorStep(4), 12000);
      } else if (colors.length === 1) {
        schedule(() => setColorStep(2), 2000);
        schedule(() => setColorStep(3), 4500);
        schedule(() => setColorStep(4), 7000);
      } else {
        schedule(() => setColorStep(3), 2000);
        schedule(() => setColorStep(4), 4000);
      }
    }
  }, [addLog, mode, schedule, applyColorResult, colors.length]);

  const goToTransition = useCallback(() => {
    if (showAct1Transition) { setCurrentStage(STAGES.TRANSITION); addLog("入境：一封来自千年前的邀请"); }
    else { goToColor(); }
  }, [addLog, showAct1Transition, goToColor]);
  useEffect(() => { goToTransitionRef.current = goToTransition; }, [goToTransition]);

  const goToDraw = useCallback(() => {
    if (!colorConfirmedRef.current) {
      console.log("[App] goToDraw blocked — 等待后端 color_confirmed");
      return;
    }
    console.log("[App] goToDraw → Act3");
    setCurrentStage(STAGES.DRAW); setIsAutoAdvancing(false);
    addLog("择色完成，进入第二幕：筑景");
  }, [addLog]);

  const goToSpirit = useCallback(() => {
    console.log("[App] goToSpirit → Act4");
    setCurrentStage(STAGES.SPIRIT); setSpiritStatus("searching");
    setWaitingForStamp(false);
    addLog("筑景完成，进入第三幕：唤灵");
  }, [addLog]);
  useEffect(() => { goToSpiritRef.current = goToSpirit; }, [goToSpirit]);

  const goToPostcard = useCallback(() => {
    console.log("[App] goToPostcard → Act5");
    postcardEnterTimeRef.current = Date.now();
    if (mode === "demo" && !narrative) {
      setNarrative(mockGenerateNarrative({ color: colors[0], objectResult, matchedCharacter }));
    }
    setCurrentStage(STAGES.POSTCARD); setWaitingForStamp(false);
    addLog("唤灵完成，进入第四幕：成色");
  }, [addLog, mode, narrative, colors, objectResult, matchedCharacter]);

  // ── Reset ──
  const resetAll = useCallback(() => {
    clearTimers();
    setCurrentStage(STAGES.INTRO); setColors([]); setColorSource(null); setGesture(null);
    setPoints([]); setObjectResult(null); setImageryItems([]);
    setMatchedCharacter(null); setSpiritStatus("idle"); setNarrative(null);
    setIsAutoAdvancing(false); setColorStep(1); setWaitingForStamp(false);
    setIsDetecting(false); setColorRound(1);
    setFirstColor(null); setSecondColor(null);
    setStableColorName(null); setStableSeconds(0);
    fistTriggeredRef.current = false; colorLockedRef.current = false;
    colorConfirmedRef.current = false;
    liveObjectCountRef.current = 0;
    usedObjectNamesRef.current = [];
    resetCooldownUntil.current = Date.now() + 3000; // 3s grace after reset
    setAct3Overlay(null);
    if (fallbackTimerRef.current) { clearTimeout(fallbackTimerRef.current); fallbackTimerRef.current = null; }
  }, [clearTimers]);

  // ── Act3 sketch recognition ──
  const recognizeSketch = useCallback(async (payload) => {
    console.log("[App] recognizeSketch — mode:", mode, "cached:", latestRef.current.objectResult?.name);
    if (mode === "demo") {
      const validObjects = ["石桥","古树","书卷","岳麓书院","湘江","爱晚亭","碑刻","竹林","讲堂","石阶","岳麓山","长廊","东方红广场","中国书院博物馆","书架","书案","匾额","古籍","图书馆","墨锭","学位帽","实验室","屋脊","山石","操场","教学楼","显微镜","林荫道","校徽","校门","楹联","毛笔","湖南大学大礼堂","牌楼路","白鹤泉","砚台","窗格","竹简","笔记本","线装书","经卷","自卑亭","荣誉证书","设计院楼","赫曦台","院墙","麓山南路","黑板"];
      const label = validObjects[Math.floor(Math.random() * validObjects.length)];
      const result = { name: label, reason: `你画下了${label}。`, stylizedImageUrl: `/src/assets/act3/objects/${label}.png` };
      setObjectResult(result);
      return { label: result.name, description: [result.reason], stylizedImageUrl: result.stylizedImageUrl };
    }
    const cached = latestRef.current.objectResult;
    if (cached) return { label: cached.name, description: [cached.reason || `你画下了${cached.name}。`], stylizedImageUrl: cached.stylizedImageUrl || `/src/assets/act3/objects/${cached.name}.png` };
    await new Promise(r => setTimeout(r, 1500));
    const retry = latestRef.current.objectResult;
    if (retry) return { label: retry.name, description: [retry.reason || `你画下了${retry.name}。`], stylizedImageUrl: retry.stylizedImageUrl || `/src/assets/act3/objects/${retry.name}.png` };
    return { label: "石桥", description: ["你画下了一个意象。"], stylizedImageUrl: "/src/assets/act3/objects/石桥.png" };
  }, [mode]);

  // ── Act4 spirit match ──
  const fetchSpiritMatch = useCallback(async () => {
    if (mode === "demo") {
      const character = mockMatchCharacter(colors[0], objectResult);
      setMatchedCharacter(character);
      return {
        person: { id: character?.name || "zhang-shi", name: character?.name || "张栻", subtitle: [character?.title || "岳麓书院讲学者"], portraitUrl: "" },
        narrative: {
          centerStart: ["颜色已经展开", "意象也已经落下。"], centerSeek: ["现在", "我要在千年的文脉里", "寻找一个与你相遇的人。"],
          loading: ["正在寻找回应你的人……"], found: ["找到了！"],
          rightInterim: ["他还不能告诉你名字", "你要先听他说完。"],
          leftBlue: character?.monologue || [], leftYellow: character?.monologue || [],
          rightFinal: character?.spiritLine ? [character.spiritLine] : ["刚才与你说话的，是他。", "但他留下的不只是名字，", "更是一种敢于发问的底色。"],
        },
      };
    }
    const ch = latestRef.current.matchedCharacter;
    if (ch) {
      console.log("[fetchSpiritMatch] character:", ch.name, "portrait:", ch.portrait);
      return {
        person: { id: ch.name, name: ch.name, subtitle: [ch.title || ""], portraitUrl: ch.portrait ? `/src/assets/act5/people/${ch.portrait}.png` : "" },
        narrative: { centerStart: ["颜色已经展开", "意象也已经落下。"], centerSeek: ["现在", "我要在千年的文脉里", "寻找一个与你相遇的人。"], loading: ["正在寻找回应你的人……"], found: ["找到了！"], rightInterim: ["他还不能告诉你名字", "你要先听他说完。"], leftBlue: ch.monologue || [], leftYellow: ch.monologue || [], rightFinal: ch.spiritLine ? [ch.spiritLine] : [] }
      };
    }
    return null;
  }, [mode, colors, objectResult]);

  // ── WebSocket message handler ──
  const handleBackendPayload = useCallback((payload) => {
    const message = normalizeBackendMessage(payload);
    if (!message) return;
    switch (message.type) {
      case MESSAGE_TYPES.COLOR_DETECTION_ACTIVE:
        setIsDetecting(true);
        setStableColorName(null);
        setStableSeconds(0);
        setColorRound(message.round || 1);
        break;

      case MESSAGE_TYPES.COLOR_DETECT_PROGRESS:
        if (message.round) setColorRound(message.round);
        setStableColorName(message.stableColor || null);
        setStableSeconds(message.elapsed || 0);
        break;

      case MESSAGE_TYPES.COLOR_DETECTED:
        if (Date.now() < resetCooldownUntil.current) break;

        // ── 新 Act2 两色流程 ──
        if (message.source === "object" || message.source === "clothing") {
          const found = findColor(message.colorName);
          const round = message.round || 1;
          if (!found) break;

          if (round === 1) {
            setFirstColor({ hex: found.hex, name: found.name });
            setIsDetecting(false);  // 停检测，等用户再按 R
            setColors([found]);
            addLog(`第一色完成：${found.name}，等待第二色…`);
          } else {
            setSecondColor({ hex: found.hex, name: found.name });
            setIsDetecting(false);
            setColors(prev => [...prev, found]);
            colorConfirmedRef.current = true;
            addLog(`第二色完成：${found.name}，两色齐备`);
          }
          break;
        }

        // ── 旧流程兼容（仅在两色均未通过新流程确认时才运行）──
        if (message.source === "confirmed") {
          colorConfirmedRef.current = true;
          // 新流程已设置好两色 → 只需要推进到 Act3
          if (secondColor) {
            if (stageRef.current === STAGES.COLOR) {
              setTimeout(() => goToDraw(), 500);
            }
            break;
          }
          if (stageRef.current === STAGES.COLOR) {
            setTimeout(() => goToDraw(), 500);
          }
        }
        // 新流程已处理过一轮以上 → 不要再让旧流程改颜色
        if (firstColor) break;
        // Only process color when on COLOR or DRAW stage or later (not during Act1)
        if (stageRef.current === STAGES.INTRO || stageRef.current === STAGES.TRANSITION) {
          const detected = findColor(message.colorName);
          setColors(prev => {
            if (prev.length >= 2) return prev;
            if (!prev.some(c => c.name === detected.name)) return [...prev, detected];
            const allColors = ["岳麓绿","书院红","湘江蓝","西迁黄","校徽金","墨色"];
            const unused = allColors.filter(n => n !== detected.name && !prev.some(c => c.name === n));
            const fbName = unused[Math.floor(Math.random() * unused.length)];
            addLog(`第二色与第一色相同，自动备选：${fbName}`);
            return [...prev, findColor(fbName)];
          });
          break;
        }
        if (colors.length < 2) {
          applyColorResult(findColor(message.colorName), message.source, message.confidence);
        }
        break;

      case MESSAGE_TYPES.GESTURE_STATE:
        setGesture(message.gesture || message.mode);
        // Live mode: fist on INTRO → start flow (once, respect cooldown)
        if (message.gesture === "fist" && stageRef.current === STAGES.INTRO && !fistTriggeredRef.current && Date.now() > resetCooldownUntil.current) {
          fistTriggeredRef.current = true;
          addLog("检测到握拳手势，开始入境");
          goToTransitionRef.current?.();
        }
        // Live mode: fist on POSTCARD → restart (after min 10s to allow viewing)
        if (message.gesture === "fist" && stageRef.current === STAGES.POSTCARD) {
          if (Date.now() - postcardEnterTimeRef.current > 10000) {
            addLog("握拳重启");
            resetAll();
          }
        }
        // Live mode: any hand gesture on SPIRIT with waitingForStamp → advance to postcard
        if (stageRef.current === STAGES.SPIRIT && waitingForStamp) {
          if (message.gesture === "fist" || message.gesture === "open_hand" || message.gesture === "index_pointing") {
            addLog("握拳盖章，生成明信片");
            goToPostcard();
          }
        }
        break;

      case MESSAGE_TYPES.DRAWING_POINT:
        // Coordinates are normalized (0-1), scale to viewport for canvas rendering
        const vw = window.innerWidth, vh = window.innerHeight;
        remoteDrawRef.current?.(message.x * vw, message.y * vh);
        // Accumulate normalized positions for bbox calculation
        drawPositionsRef.current.push({ x: message.x, y: message.y });
        break;

      case MESSAGE_TYPES.OBJECT_RECOGNIZED: {
        liveObjectCountRef.current += 1;
        // Dedup object name: use ref (state may be stale)
        let objName = message.name;
        if (usedObjectNamesRef.current.includes(objName)) {
          const fallbacks = ["古树","书卷","石阶","岳麓书院","竹简","碑刻","讲堂","爱晚亭","石桥","长廊","匾额","竹林","东方红广场","古籍","线装书","经卷","墨锭","砚台","毛笔","窗格","院墙","屋脊","自卑亭","赫曦台","校门","林荫道","湘江","白鹤泉","山石","图书馆","书架","书案","学位帽","校徽","荣誉证书","笔记本","黑板"];
          const unused = fallbacks.filter(n => !usedObjectNamesRef.current.includes(n));
          objName = unused[Math.floor(Math.random() * unused.length)];
          console.log("[App] Object dedup: same name, fallback to", objName);
        }
        // Calculate position from accumulated drawing points
        const pts = drawPositionsRef.current;
        let position = null;
        if (pts.length > 3) {
          const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
          const minX = Math.min(...xs), maxX = Math.max(...xs);
          const minY = Math.min(...ys), maxY = Math.max(...ys);
          // 坐标已归一化到 0-1，直接转百分比
          position = { left: (minX + maxX) / 2 * 100, top: (minY + maxY) / 2 * 100, width: Math.max((maxX - minX) * 100, 10) };
        }
        drawPositionsRef.current = []; // reset for next drawing
        remoteDrawRef.current?.clear?.(); // clear remote canvas for next drawing
        usedObjectNamesRef.current.push(objName);
        const pointillistUrl = `/src/assets/act3/objects/${objName}.png`;
        console.log("[App] OBJECT_RECOGNIZED:", objName, "count:", liveObjectCountRef.current, "pos:", position, "img:", pointillistUrl);
        setObjectResult({ ...message, name: objName, stylizedImageUrl: pointillistUrl });
        setImageryItems(prev => [...prev, {
          id: objName + liveObjectCountRef.current,
          name: objName,
          description: [message.reason || `你画下了${objName}。`],
          imageUrl: pointillistUrl,
          className: prev.length === 0 ? "act5__imagery--bridge" : "act5__imagery--tree",
          position,
        }]);
        addLog(`筑景完成：线条被叙事化为${objName}`);
        if (mode === "live" && liveObjectCountRef.current >= 2) {
          console.log("[App] 2 objects confirmed, auto-advancing to spirit in 4s");
          addLog("两个意象已经落下，4秒后进入唤灵");
          setTimeout(() => goToSpiritRef.current?.(), 4000);
        }
        break;
      }

      case MESSAGE_TYPES.CHARACTER_MATCHED:
      case MESSAGE_TYPES.CHARACTERS_RECOMMENDED:
        if (message.character) {
          console.log("[App] CHARACTER stored:", message.character.name, "portrait:", message.character.portrait);
          // Merge with existing — character_revealed may not have monologue/spiritLine
          setMatchedCharacter(prev => ({ ...(prev || {}), ...message.character,
            monologue: message.character.monologue?.length ? message.character.monologue : (prev?.monologue || []),
            spiritLine: message.character.spiritLine || prev?.spiritLine || "",
            portrait: message.character.portrait || prev?.portrait || "",
          }));
          addLog(`人物匹配：${message.character.name}`);
        }
        break;

      case MESSAGE_TYPES.NARRATIVE_GENERATED:
        console.log("[App] NARRATIVE stored:", message.title);
        setNarrative(message);
        addLog("收到 AI 叙事");
        // Safety net: if still stuck on Act3, auto-advance to spirit
        if (stageRef.current === STAGES.DRAW && mode === "live") {
          console.log("[App] Narrative arrived while on DRAW — force advancing to spirit");
          setTimeout(() => goToSpiritRef.current?.(), 500);
        }
        break;

      case MESSAGE_TYPES.POSTCARD_READY:
        setPostcardQr(message.qrBase64 || "");
        setPostcardImageUrl(message.imageUrl || "");
        addLog(`明信片已生成，扫码下载: ${message.imageUrl}`);
        break;

      case MESSAGE_TYPES.SYSTEM_LOG:
        addLog(`[后端 ${message.level}] ${message.message}`);
        break;
    }
  }, [addLog, applyColorResult, colors, imageryItems.length, resetAll, waitingForStamp, goToPostcard]);

  const socket = useWebSocket(WS_URL, handleBackendPayload);
  const wsSend = socket.send;
  useEffect(() => { socket.connect(); }, []); // auto-connect in live mode

  // Capture Act3 scene as overlay for Act4/Act5
  const onAct3Snapshot = useCallback((dataUrl) => { setAct3Overlay(dataUrl); }, []);

  // ── Act3 imagery confirmed callback (captures bbox for postcard) ──
  const onImageryConfirmed = useCallback((item) => {
    console.log("[App] imagery confirmed:", item.label, "bbox:", item.bbox?.normalized);
    setImageryItems(prev => [...prev, {
      id: item.label,
      name: item.label,
      description: item.description || [],
      imageUrl: item.stylizedImageUrl || "",
      className: prev.length === 0 ? "act5__imagery--bridge" : "act5__imagery--tree",
      position: item.bbox?.normalized ? {
        left: item.bbox.normalized.centerX * 100,
        top: item.bbox.normalized.centerY * 100,
        width: item.bbox.normalized.width * 100,
      } : null,
    }]);
    // Live mode: auto-advance after 2 objects
    if (mode === "live" && imageryItems.length + 1 >= 2) {
      console.log("[App] 2 objects confirmed, auto-advancing to spirit");
      schedule(() => goToSpiritRef.current?.(), 2000);
    }
  }, [mode, imageryItems.length, schedule]);

  // ── Act4 onComplete — show stamp prompt instead of auto-advancing ──
  const onAct4Complete = useCallback(() => {
    if (mode === "live") {
      setWaitingForStamp(true);
      addLog("握拳盖章，完成你的千年色");
    } else {
      goToPostcard();
    }
  }, [mode, goToPostcard, addLog]);

  // ── Render ──
  const actualColors = colors.length > 0 ? colors : [];

  function renderCurrentStage() {
    switch (currentStage) {
      case STAGES.INTRO:
        return <Act0 onNext={goToTransition} autoAdvanceDelay={5000} waitForGesture={true} />;

      case STAGES.TRANSITION:
        return <Act1Entry switchDelay={7000} dissolveDelay={18000} onComplete={goToColor} onSkip={goToColor} dissolveOnCompleteDelay={1200} />;

      case STAGES.COLOR:
        return (
          <Act2ColorSeeking
            round={colorRound}
            firstColor={firstColor}
            secondColor={secondColor}
            isDetecting={isDetecting}
            stableColorName={stableColorName}
            stableSeconds={stableSeconds}
            confirmSeconds={confirmSeconds}
            onComplete={goToDraw}
            completeDelay={2000}
          />
        );

      case STAGES.DRAW:
        return (
          <Act3FormingVision
            primaryColor={actualColors[0]?.hex || "#F2E700"}
            secondaryColor={actualColors[1]?.hex || actualColors[0]?.hex || "#355BFF"}
            maxRounds={2}
            onRecognizeSketch={recognizeSketch}
            onImageryConfirmed={onImageryConfirmed}
            onComplete={() => { console.log("[App] Act3 onComplete"); goToSpirit(); }}
            onCanvasSnapshot={onAct3Snapshot}
            completeDelay={3000}
            liveConfirmedItems={imageryItems.map(it => ({ id: it.id, label: it.name, description: it.description, stylizedImageUrl: it.imageUrl, overlay: { scale: 1.35, offsetX: 0, offsetY: 0 }, bbox: { normalized: { width: it.position?.width ? it.position.width / 100 : 0.2, height: 0.2, centerX: it.position?.left ? it.position.left / 100 : 0.5, centerY: it.position?.top ? it.position.top / 100 : 0.5 } } }))}
            remotePoints={[]}
            remoteDrawRef={remoteDrawRef}
          />
        );

      case STAGES.SPIRIT: {
        const a4 = buildAct4Payload(actualColors, objectResult);
        return (
          <Act4SpiritCalling
            key={`${matchedCharacter?.name || "act4"}_${matchedCharacter?.portrait || "nopic"}`}
            {...a4}
            onFetchSpiritMatch={fetchSpiritMatch}
            waitingForStamp={waitingForStamp}
            onComplete={onAct4Complete}
            completeDelay={5000}
          />
        );
      }

      case STAGES.POSTCARD:
        return (
          <Act5Postcard
            postcardData={buildAct5Data(actualColors, imageryItems, matchedCharacter, narrative, postcardQr, postcardImageUrl)}
            autoPlay={true}
            onComplete={() => addLog("明信片生成完成")}
            wsSend={wsSend}
          />
        );

      default:
        return <Act0 onNext={goToTransition} autoAdvanceDelay={4000} waitForGesture={true} />;
    }
  }

  return (
    <div className="min-h-screen bg-ink text-white">
      <main>
        {renderCurrentStage()}
        {/* Persistent Act3 overlay — keep drawing lines visible through Act4/Act5 */}
        {act3Overlay && (currentStage === STAGES.SPIRIT || currentStage === STAGES.POSTCARD) && (
          <img src={act3Overlay} style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh", objectFit: "contain", pointerEvents: "none", zIndex: 999, opacity: 0.3 }} alt="" />
        )}
      </main>
    </div>
  );
}
