import { useCallback, useEffect, useRef, useState } from "react";
import Header from "./components/Header";
import StageStepper from "./components/StageStepper";
import StatusPanel from "./components/StatusPanel";
import SystemLog from "./components/SystemLog";
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
    aiWriting: narrative?.paragraphs || ["你的千年色正在成形……"],
    person: { name: matchedCharacter?.name || "回应者", portraitUrl: "" },
    mainTitleImageUrl: imageUrl || "", downloadQrUrl: qrBase64 || "",
    createdAtText: dateStr,
  };
}

export default function App() {
  const [mode, setMode] = useState("demo");
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
  const [logs, setLogs] = useState([]);
  const [isAutoAdvancing, setIsAutoAdvancing] = useState(false);
  const [colorStep, setColorStep] = useState(1);
  const [showAct1Transition, setShowAct1Transition] = useState(true);
  const [waitingForStamp, setWaitingForStamp] = useState(false); // Act4 等握拳盖章
  const [postcardQr, setPostcardQr] = useState("");       // QR base64 from backend
  const [postcardImageUrl, setPostcardImageUrl] = useState(""); // postcard image URL
  const postcardEnterTimeRef = useRef(0);    // when Act5 was entered

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

  useEffect(() => { colorsRef.current = colors; }, [colors]);

  useEffect(() => { latestRef.current = { mode, colors, objectResult, matchedCharacter, narrative, imageryItems }; }, [mode, colors, objectResult, matchedCharacter, narrative, imageryItems]);
  useEffect(() => { stageRef.current = currentStage; }, [currentStage]);
  useEffect(() => { showAct1TransitionRef.current = showAct1Transition; }, [showAct1Transition]);

  const clearTimers = useCallback(() => { timersRef.current.forEach(clearTimeout); timersRef.current = []; }, []);
  const schedule = useCallback((cb, delay) => { const t = setTimeout(() => { timersRef.current = timersRef.current.filter(i => i !== t); cb(); }, delay); timersRef.current.push(t); return t; }, []);
  useEffect(() => clearTimers, [clearTimers]);

  const addLog = useCallback((msg) => { setLogs(c => [...c, { id: `${Date.now()}-${Math.random()}`, time: new Date().toLocaleTimeString("zh-CN", { hour12: false }), message: msg }]); }, []);

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
        schedule(() => setColorStep(2), 2500);
        schedule(() => setColorStep(3), 5000);
        schedule(() => setColorStep(4), 7500);
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
        }, 10000);
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
    // Live mode: if colors already cached (during Act1), drive full step progression
    if (mode === "live" && colors.length > 0) {
      if (colors.length >= 2) {
        // Both colors cached → fast-forward to step 4
        schedule(() => setColorStep(3), 2000);
        schedule(() => setColorStep(4), 4000);
      } else {
        // One color cached → show steps 2→3→4
        schedule(() => setColorStep(2), 2000);
        schedule(() => setColorStep(3), 4500);
        schedule(() => setColorStep(4), 7000);
      }
    }
  }, [addLog, mode, schedule, applyColorResult, colors.length]);

  const goToTransition = useCallback(() => {
    if (showAct1Transition) { setCurrentStage(STAGES.TRANSITION); addLog("入境：一封来自千年前的邀请"); }
    else { goToColor(); }
  }, [addLog, showAct1Transition, goToColor]);
  useEffect(() => { goToTransitionRef.current = goToTransition; }, [goToTransition]);

  const goToDraw = useCallback(() => {
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
    setLogs([]); setIsAutoAdvancing(false); setColorStep(1); setWaitingForStamp(false);
    fistTriggeredRef.current = false; colorLockedRef.current = false;
    liveObjectCountRef.current = 0;
    usedObjectNamesRef.current = [];
    resetCooldownUntil.current = Date.now() + 3000; // 3s grace after reset
    if (fallbackTimerRef.current) { clearTimeout(fallbackTimerRef.current); fallbackTimerRef.current = null; }
  }, [clearTimers]);

  // ── Mode switching ──
  const handleModeChange = (nextMode) => {
    if (nextMode === mode) return;
    if (mode === "live") socket.disconnect();
    resetAll();
    setMode(nextMode);
    setLogs([{ id: `${Date.now()}-mode`, time: new Date().toLocaleTimeString("zh-CN", { hour12: false }), message: `切换至 ${nextMode === "demo" ? "Demo" : "Live"} Mode` }]);
  };

  // ── Act3 sketch recognition ──
  const recognizeSketch = useCallback(async (payload) => {
    console.log("[App] recognizeSketch — mode:", mode, "cached:", latestRef.current.objectResult?.name);
    if (mode === "demo") {
      const result = mockRecognizeObject(points.length > 0 ? points : [{ x: 200, y: 200 }, { x: 400, y: 200 }], colors[0]);
      setObjectResult(result);
      return { label: result.name, description: [result.reason] };
    }
    const cached = latestRef.current.objectResult;
    if (cached) return { label: cached.name, description: [cached.reason || `你画下了${cached.name}。`] };
    await new Promise(r => setTimeout(r, 1500));
    const retry = latestRef.current.objectResult;
    if (retry) return { label: retry.name, description: [retry.reason || `你画下了${retry.name}。`] };
    return { label: "桥", description: ["你画下了一个意象。"] };
  }, [mode, points, colors]);

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
    if (ch) return { person: { id: ch.name, name: ch.name, subtitle: [ch.title || ""], portraitUrl: "" }, narrative: { centerStart: ["颜色已经展开", "意象也已经落下。"], centerSeek: ["现在", "我要在千年的文脉里", "寻找一个与你相遇的人。"], loading: ["正在寻找回应你的人……"], found: ["找到了！"], rightInterim: ["他还不能告诉你名字", "你要先听他说完。"], leftBlue: ch.monologue || [], leftYellow: ch.monologue || [], rightFinal: ch.spiritLine ? [ch.spiritLine] : [] } };
    return null;
  }, [mode, colors, objectResult]);

  // ── WebSocket message handler ──
  const handleBackendPayload = useCallback((payload) => {
    const message = normalizeBackendMessage(payload);
    if (!message) return;
    switch (message.type) {
      case MESSAGE_TYPES.COLOR_DETECTED:
        // Cooldown after reset: ignore rapid color messages
        if (Date.now() < resetCooldownUntil.current) break;
        // Only process color when on COLOR or DRAW stage or later (not during Act1)
        if (stageRef.current === STAGES.INTRO || stageRef.current === STAGES.TRANSITION) {
          // Cache color silently with dedup+fallback
          const detected = findColor(message.colorName);
          setColors(prev => {
            if (prev.length >= 2) return prev;
            if (!prev.some(c => c.name === detected.name)) return [...prev, detected];
            // Same color — pick different fallback
            const allColors = ["岳麓绿","书院红","湘江蓝","西迁黄","校徽金","墨色"];
            const unused = allColors.filter(n => n !== detected.name && !prev.some(c => c.name === n));
            const fbName = unused[Math.floor(Math.random() * unused.length)];
            addLog(`第二色与第一色相同，自动备选：${fbName}`);
            return [...prev, findColor(fbName)];
          });
          break;
        }
        // Process normally if already on Act2+
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
        // Live mode: fist on SPIRIT with waitingForStamp → advance to postcard
        if (message.gesture === "fist" && stageRef.current === STAGES.SPIRIT && waitingForStamp) {
          addLog("握拳盖章，生成明信片");
          goToPostcard();
        }
        break;

      case MESSAGE_TYPES.DRAWING_POINT:
        // Direct canvas rendering — bypass React state entirely for performance
        remoteDrawRef.current?.(message.x, message.y);
        // Still accumulate positions for bbox calculation (cheap, no render)
        drawPositionsRef.current.push({ x: message.x, y: message.y });
        break;

      case MESSAGE_TYPES.OBJECT_RECOGNIZED: {
        liveObjectCountRef.current += 1;
        // Dedup object name: use ref (state may be stale)
        let objName = message.name;
        if (usedObjectNamesRef.current.includes(objName)) {
          const fallbacks = ["古树","竹林","桥","山","灯","鸟","书卷","讲堂","亭台"];
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
          position = { left: (minX + maxX) / 2 / 1280 * 100, top: (minY + maxY) / 2 / 720 * 100, width: Math.max((maxX - minX) / 1280 * 100, 10) };
        }
        drawPositionsRef.current = []; // reset for next drawing
        remoteDrawRef.current?.clear?.(); // clear remote canvas for next drawing
        usedObjectNamesRef.current.push(objName);
        console.log("[App] OBJECT_RECOGNIZED:", objName, "count:", liveObjectCountRef.current, "pos:", position);
        setObjectResult({ ...message, name: objName });
        setImageryItems(prev => [...prev, {
          id: objName + liveObjectCountRef.current,
          name: objName,
          description: [message.reason || `你画下了${objName}。`],
          imageUrl: "",
          className: prev.length === 0 ? "act5__imagery--bridge" : "act5__imagery--tree",
          position,
        }]);
        addLog(`筑景完成：线条被叙事化为${objName}`);
        if (mode === "live" && liveObjectCountRef.current >= 2) {
          console.log("[App] 2 objects confirmed, auto-advancing to spirit");
          addLog("两个意象已经落下，进入唤灵");
          setTimeout(() => goToSpiritRef.current?.(), 1500);
        }
        break;
      }

      case MESSAGE_TYPES.CHARACTER_MATCHED:
      case MESSAGE_TYPES.CHARACTERS_RECOMMENDED:
        if (message.character) {
          console.log("[App] CHARACTER stored:", message.character.name);
          setMatchedCharacter(message.character);
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
  const isActPage = currentStage !== "intro_old";
  const actualColors = colors.length > 0 ? colors : [];

  function renderCurrentStage() {
    switch (currentStage) {
      case STAGES.INTRO:
        return <Act0 onNext={goToTransition} autoAdvanceDelay={5000} waitForGesture={mode === "live"} />;

      case STAGES.TRANSITION:
        return <Act1Entry switchDelay={7000} dissolveDelay={18000} onComplete={goToColor} onSkip={goToColor} dissolveOnCompleteDelay={1200} />;

      case STAGES.COLOR:
        return (
          <Act2ColorSeeking
            step={colorStep}
            recognizedColors={actualColors.map(c => c.hex)}
            copyByStep={buildAct2CopyByStep(actualColors)}
            autoDemo={false}
            stepDuration={6000}
            onComplete={goToDraw}
            completeDelay={4000}
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
            completeDelay={3000}
            remotePoints={[]}
            remoteDrawRef={remoteDrawRef}
          />
        );

      case STAGES.SPIRIT: {
        const a4 = buildAct4Payload(actualColors, objectResult);
        return (
          <Act4SpiritCalling
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
            onRestart={mode === "demo" ? resetAll : undefined}
          />
        );

      default:
        return <Act0 onNext={goToTransition} autoAdvanceDelay={4000} />;
    }
  }

  return (
    <div className="min-h-screen bg-ink text-white">
      <div className={isActPage ? "fixed top-0 left-0 right-0 z-50 pointer-events-none" : ""}>
        <div className={isActPage ? "pointer-events-auto" : ""}>
          <Header mode={mode} onModeChange={handleModeChange} wsStatus={socket.status} wsError={socket.error} onConnect={socket.connect} onDisconnect={socket.disconnect} />
        </div>
      </div>
      {!isActPage && <div className="border-b border-white/5 bg-black/10 px-4 py-4"><StageStepper currentStage={currentStage} /></div>}
      <main className={isActPage ? "" : "mx-auto max-w-[1280px] px-4 py-6 md:px-8"}>
        {renderCurrentStage()}
        {(mode === "live" || !isActPage) && (
          <details className={mode === "live" && isActPage ? "fixed bottom-4 right-4 z-[100] max-w-sm rounded-xl border border-white/10 bg-black/85 text-sm text-white/60 backdrop-blur" : "mx-auto mt-6 max-w-5xl rounded-xl border border-white/5 bg-black/10 text-sm text-white/45"}>
            <summary className="cursor-pointer px-4 py-3 hover:text-white/65">查看系统状态与事件日志</summary>
            <div className="grid gap-5 border-t border-white/5 p-4 md:grid-cols-2">
              <div><p className="eyebrow mb-3">CURRENT STATE</p><StatusPanel currentStage={currentStage} color={actualColors[0]} colorSource={colorSource} gesture={gesture} objectResult={objectResult} matchedCharacter={matchedCharacter} isAutoAdvancing={isAutoAdvancing} /></div>
              <div><p className="eyebrow mb-3">SYSTEM LOG</p><SystemLog logs={logs} /></div>
            </div>
          </details>
        )}
      </main>
    </div>
  );
}
