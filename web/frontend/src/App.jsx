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

// ── Act pages ──
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

// ── Stage constants ──
const STAGES = {
  INTRO: "intro",
  TRANSITION: "transition",
  COLOR: "color",
  DRAW: "draw",
  SPIRIT: "spirit",
  POSTCARD: "postcard",
};

// ── Data bridging: App state → Act props ──

function buildAct2CopyByStep(color) {
  if (!color) {
    return {
      1: ["请将随身之物靠近光中。", "让它替你说话。"],
      2: ["让我看看……", "这件东西里", "藏着怎样的颜色。"],
      3: [],
      4: [],
    };
  }
  return {
    1: ["请将随身之物靠近光中。", "让它替你说话。"],
    2: ["让我看看……", "这件东西里", "藏着怎样的颜色。"],
    3: [`一缕${color.name}浮出光面，`, `像旧纸上醒来的日色。`],
    4: [`${color.name}与另一种颜色相遇，`, `像山门灯火照见夜色。`],
  };
}

function buildAct4Payload(color, objectResult) {
  return {
    primaryColor: color?.hex || "#F2E700",
    secondaryColor: color?.hex || "#355BFF",
    firstColorName: color?.name || "黄",
    secondColorName: color?.name || "蓝",
    firstImageryName: objectResult?.name || "桥",
    secondImageryName: objectResult?.name || "树",
  };
}

function buildAct5Data(color, objectResult, matchedCharacter, narrative) {
  const now = new Date();
  const dateStr = `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, "0")}.${String(now.getDate()).padStart(2, "0")}\n${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  return {
    colors: {
      primary: color?.hex || "#F2E700",
      secondary: "#355BFF",
      primaryName: color?.name || "明黄",
      secondaryName: "澄蓝",
    },
    selectedPlaces: ["岳麓山", "湘江水", "书院檐角"],
    title: {
      cn: narrative?.title || `${color?.name || "千年色"} · ${objectResult?.name || "湖大"}之境`,
      en: "THE MILLENNIUM COLOR OF YOURS",
    },
    traceText: `[${color?.name || "?"}] → [${objectResult?.name || "?"}] → [${matchedCharacter?.name || "?"}]`,
    imageryItems: [
      {
        id: "object",
        name: objectResult?.name || "桥",
        imageUrl: "",
        className: "act5__imagery--bridge",
      },
      {
        id: "tree",
        name: "树",
        imageUrl: "",
        className: "act5__imagery--tree",
      },
    ],
    objectText: objectResult?.reason
      ? [objectResult.reason]
      : ["这个意象已经落下。"],
    aiWriting: narrative?.paragraphs || ["你的千年色正在成形……"],
    person: {
      name: matchedCharacter?.name || "回应者",
      portraitUrl: "",
    },
    mainTitleImageUrl: "",
    downloadQrUrl: "",
    createdAtText: dateStr,
  };
}

export default function App() {
  const [mode, setMode] = useState("demo");
  const [currentStage, setCurrentStage] = useState(STAGES.INTRO);
  const [color, setColor] = useState(null);
  const [colorSource, setColorSource] = useState(null);
  const [gesture, setGesture] = useState(null);
  const [points, setPoints] = useState([]);
  const [objectResult, setObjectResult] = useState(null);
  const [matchedCharacter, setMatchedCharacter] = useState(null);
  const [spiritStatus, setSpiritStatus] = useState("idle");
  const [narrative, setNarrative] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isAutoAdvancing, setIsAutoAdvancing] = useState(false);
  const [colorStep, setColorStep] = useState(1);
  const [showAct1Transition, setShowAct1Transition] = useState(true);

  const timersRef = useRef([]);
  const latestRef = useRef({ mode, color, objectResult, matchedCharacter });
  const stageRef = useRef(currentStage);
  const goToTransitionRef = useRef(null);
  const showAct1TransitionRef = useRef(showAct1Transition);

  useEffect(() => {
    latestRef.current = { mode, color, objectResult, matchedCharacter };
  }, [mode, color, objectResult, matchedCharacter]);
  useEffect(() => { stageRef.current = currentStage; }, [currentStage]);
  useEffect(() => { showAct1TransitionRef.current = showAct1Transition; }, [showAct1Transition]);

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  const schedule = useCallback((callback, delay) => {
    const timer = setTimeout(() => {
      timersRef.current = timersRef.current.filter((item) => item !== timer);
      callback();
    }, delay);
    timersRef.current.push(timer);
    return timer;
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  const addLog = useCallback((message) => {
    setLogs((current) => [
      ...current,
      {
        id: `${Date.now()}-${Math.random()}`,
        time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
        message,
      },
    ]);
  }, []);

  // ── Apply callbacks (same for demo + live) ──

  const applyColorResult = useCallback((detectedColor, source, confidence = null) => {
    clearTimers();
    setCurrentStage(STAGES.COLOR);
    setColor(detectedColor);
    setColorSource(
      confidence === null
        ? source
        : `${source} · ${Math.round(confidence * 100)}%`,
    );
    setPoints([]);
    setObjectResult(null);
    setMatchedCharacter(null);
    setSpiritStatus("idle");
    setNarrative(null);
    setIsAutoAdvancing(true);
    addLog(`择色完成：读取为${detectedColor.name}`);

    // Drive Act2 step progression (slowed down for pacing)
    setColorStep(2);
    schedule(() => setColorStep(3), 2500);
    schedule(() => setColorStep(4), 5000);
    // Act2's onComplete will advance to draw
  }, [addLog, clearTimers, schedule]);

  const applyObjectResult = useCallback((recognizedObject) => {
    clearTimers();
    setCurrentStage(STAGES.DRAW);
    setObjectResult(recognizedObject);
    setGesture("drawing_complete");
    setIsAutoAdvancing(true);
    addLog(`筑景完成：线条被叙事化为${recognizedObject.name}`);

    schedule(() => {
      beginSpiritReveal();
    }, 1500);
  }, [addLog, clearTimers, schedule]);

  // ── Spirit reveal (Act4 handles its own timeline; we just set stage + data) ──

  const beginSpiritReveal = useCallback((character = null) => {
    clearTimers();
    const latest = latestRef.current;
    const resolvedCharacter = character
      ?? (latest.mode === "demo" && latest.color && latest.objectResult
        ? mockMatchCharacter(latest.color, latest.objectResult)
        : latest.matchedCharacter);

    setCurrentStage(STAGES.SPIRIT);
    setMatchedCharacter(resolvedCharacter ?? null);
    setSpiritStatus("searching");
    setIsAutoAdvancing(false);
    addLog("进入第三幕：正在千年文脉中寻找回应");
  }, [addLog, clearTimers]);

  // ── WebSocket message handler ──

  const handleBackendPayload = useCallback((payload) => {
    const message = normalizeBackendMessage(payload);
    if (!message) {
      console.log("[App] 无法识别:", payload?.type, payload);
      return;
    }
    console.log("[App] 收到:", message.type, message);
    switch (message.type) {
      case MESSAGE_TYPES.COLOR_DETECTED:
        // Color detection DRIVES stage transitions (Act2 step progression)
        applyColorResult(
          findColor(message.colorName),
          message.source,
          message.confidence,
        );
        break;

      case MESSAGE_TYPES.GESTURE_STATE:
        console.log("[App] GESTURE_STATE — gesture:", message.gesture, "mode:", message.mode, "currentStage:", stageRef.current, "goToTransition ready:", !!goToTransitionRef.current);
        setGesture(message.gesture || message.mode);
        // Live mode: fist gesture triggers flow start from intro screen
        if (message.gesture === "fist" && stageRef.current === STAGES.INTRO) {
          console.log("[App] FIST detected on INTRO — triggering goToTransition");
          addLog("检测到握拳手势，开始入境");
          goToTransitionRef.current?.();
        } else if (message.gesture === "fist") {
          console.log("[App] FIST detected but stage is", stageRef.current, "(not INTRO)");
        }
        break;

      case MESSAGE_TYPES.DRAWING_POINT:
        // Accumulate points for Act3 to render; don't jump stage
        setPoints((current) => [...current, { x: message.x, y: message.y }]);
        break;

      case MESSAGE_TYPES.OBJECT_RECOGNIZED:
        // Store object result for Act3/4/5; don't jump stage
        setObjectResult(message);
        addLog(`筑景完成：线条被叙事化为${message.name}`);
        break;

      case MESSAGE_TYPES.CHARACTER_MATCHED:
      case MESSAGE_TYPES.CHARACTERS_RECOMMENDED:
        // Store character for Act4/5; don't jump stage
        if (message.character) {
          setMatchedCharacter(message.character);
          addLog(`人物匹配：${message.character.name}`);
        } else {
          addLog("人物消息中没有可用人物数据");
        }
        break;

      case MESSAGE_TYPES.NARRATIVE_GENERATED:
        // Store narrative for Act5; don't jump stage
        setNarrative(message);
        addLog("收到 AI 叙事");
        break;

      case MESSAGE_TYPES.POSTCARD_READY:
        addLog(`明信片已生成，扫码下载: ${message.imageUrl}`);
        break;

      case MESSAGE_TYPES.SYSTEM_LOG:
        addLog(`[后端 ${message.level}] ${message.message}`);
        break;

      default:
        break;
    }
  }, [
    addLog,
    applyColorResult,
  ]);

  const socket = useWebSocket(WS_URL, handleBackendPayload);

  // ── Reset ──

  const resetAll = useCallback(() => {
    clearTimers();
    setCurrentStage(STAGES.INTRO);
    setColor(null);
    setColorSource(null);
    setGesture(null);
    setPoints([]);
    setObjectResult(null);
    setMatchedCharacter(null);
    setSpiritStatus("idle");
    setNarrative(null);
    setLogs([]);
    setIsAutoAdvancing(false);
    setColorStep(1);
  }, [clearTimers]);

  // ── Mode switching ──

  const handleModeChange = (nextMode) => {
    if (nextMode === mode) return;
    if (mode === "live") socket.disconnect();
    resetAll();
    setMode(nextMode);
    setLogs([{
      id: `${Date.now()}-mode`,
      time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
      message: `切换至 ${nextMode === "demo" ? "Demo" : "Live"} Mode`,
    }]);
  };

  // ── Stage transition handlers ──

  const goToColor = useCallback(() => {
    setCurrentStage(STAGES.COLOR);
    setColorStep(1);
    addLog("入境完成，进入寻色");
    // Demo mode: auto-trigger mock color detection after brief delay
    if (mode === "demo") {
      const result = mockDetectColor();
      schedule(() => applyColorResult(result.color, result.source, result.confidence), 3000);
    }
  }, [addLog, mode, schedule, applyColorResult]);

  const goToTransition = useCallback(() => {
    if (showAct1Transition) {
      setCurrentStage(STAGES.TRANSITION);
      addLog("入境：一封来自千年前的邀请");
    } else {
      goToColor();
    }
  }, [addLog, showAct1Transition, goToColor]);
  useEffect(() => { goToTransitionRef.current = goToTransition; }, [goToTransition]);

  const goToDraw = useCallback(() => {
    console.log("[App] goToDraw - transitioning to Act3, color:", latestRef.current.color);
    setCurrentStage(STAGES.DRAW);
    setIsAutoAdvancing(false);
    addLog("择色完成，进入第二幕：筑景");
  }, [addLog]);

  const goToSpirit = useCallback(() => {
    setCurrentStage(STAGES.SPIRIT);
    setSpiritStatus("searching");
    addLog("筑景完成，进入第三幕：唤灵");
  }, [addLog]);

  const goToPostcard = useCallback(() => {
    if (mode === "demo" && !narrative) {
      const generated = mockGenerateNarrative({ color, objectResult, matchedCharacter });
      setNarrative(generated);
    }
    setCurrentStage(STAGES.POSTCARD);
    addLog("唤灵完成，进入第四幕：成色");
  }, [addLog, mode, narrative, color, objectResult, matchedCharacter]);

  // ── Act4 onFetchSpiritMatch — returns cached data ──

  const fetchSpiritMatch = useCallback(async (payload) => {
    if (mode === "demo") {
      const character = mockMatchCharacter(color, objectResult);
      setMatchedCharacter(character);
      return {
        person: {
          id: character?.name || "zhang-shi",
          name: character?.name || "张栻",
          subtitle: [character?.title || "岳麓书院讲学者"],
          portraitUrl: "",
        },
        narrative: {
          centerStart: ["颜色已经展开", "意象也已经落下。"],
          centerSeek: ["现在", "我要在千年的文脉里", "寻找一个与你相遇的人。"],
          loading: ["正在寻找回应你的人……"],
          found: ["找到了！"],
          rightInterim: ["他还不能告诉你名字", "你要先听他说完。"],
          leftBlue: character?.monologue || [],
          leftYellow: character?.monologue || [],
          rightFinal: character?.spiritLine
            ? [character.spiritLine]
            : ["刚才与你说话的，是他。", "但他留下的不只是名字，", "更是一种敢于发问的底色。"],
        },
      };
    }
    // Live mode: return pre-cached matchedCharacter data
    const latest = latestRef.current;
    if (latest.matchedCharacter) {
      const ch = latest.matchedCharacter;
      return {
        person: {
          id: ch.name || "unknown",
          name: ch.name || "回应者",
          subtitle: [ch.title || ""],
          portraitUrl: "",
        },
        narrative: {
          centerStart: ["颜色已经展开", "意象也已经落下。"],
          centerSeek: ["现在", "我要在千年的文脉里", "寻找一个与你相遇的人。"],
          loading: ["正在寻找回应你的人……"],
          found: ["找到了！"],
          rightInterim: ["他还不能告诉你名字", "你要先听他说完。"],
          leftBlue: ch.monologue || [],
          leftYellow: ch.monologue || [],
          rightFinal: ch.spiritLine
            ? [ch.spiritLine]
            : [],
        },
      };
    }
    return null;
  }, [mode, color, objectResult]);

  // ── Act3 onRecognizeSketch — demo mode ──

  const recognizeSketch = useCallback(async (payload) => {
    if (mode === "demo") {
      const result = mockRecognizeObject(points.length > 0 ? points : [{ x: 200, y: 200 }, { x: 400, y: 200 }], color);
      setObjectResult(result);
      return {
        label: result.name,
        description: [result.reason],
      };
    }
    // Live mode: backend already sent object_recognized — use cached data
    const cached = latestRef.current.objectResult;
    if (cached) {
      return {
        label: cached.name,
        description: [cached.reason || `你画下了${cached.name}。`],
      };
    }
    // Fallback: wait a moment and check again
    await new Promise((r) => setTimeout(r, 1500));
    const retry = latestRef.current.objectResult;
    if (retry) {
      return {
        label: retry.name,
        description: [retry.reason || `你画下了${retry.name}。`],
      };
    }
    return { label: "桥", description: ["你画下了一个意象。"] };
  }, [mode, points, color]);

  // ── Render ──

  // Show debug panel during act pages in live mode (for monitoring)
  const isActPage = currentStage !== "intro_old";
  const showDebugPanel = !isActPage || mode === "live";

  function renderCurrentStage() {
    switch (currentStage) {
      case STAGES.INTRO:
        return (
          <Act0
            onNext={goToTransition}
            autoAdvanceDelay={5000}
            waitForGesture={mode === "live"}
          />
        );

      case STAGES.TRANSITION:
        return (
          <Act1Entry
            switchDelay={7000}
            dissolveDelay={18000}
            onComplete={goToColor}
            onSkip={goToColor}
            dissolveOnCompleteDelay={1200}
          />
        );

      case STAGES.COLOR:
        return (
          <Act2ColorSeeking
            step={colorStep}
            recognizedColors={color ? [color.hex, color.hex] : []}
            copyByStep={buildAct2CopyByStep(color)}
            autoDemo={false}
            stepDuration={6000}
            onComplete={goToDraw}
            completeDelay={4000}
          />
        );

      case STAGES.DRAW: {
        const act4Base = buildAct4Payload(color, objectResult);
        return (
          <Act3FormingVision
            primaryColor={color?.hex || act4Base.primaryColor}
            secondaryColor={color?.hex || act4Base.secondaryColor}
            maxRounds={2}
            onRecognizeSketch={recognizeSketch}
            onComplete={goToSpirit}
            completeDelay={3000}
            remotePoints={mode === "live" ? points : []}
          />
        );
      }

      case STAGES.SPIRIT: {
        const a4 = buildAct4Payload(color, objectResult);
        return (
          <Act4SpiritCalling
            primaryColor={a4.primaryColor}
            secondaryColor={a4.secondaryColor}
            firstColorName={a4.firstColorName}
            secondColorName={a4.secondColorName}
            firstImageryName={a4.firstImageryName}
            secondImageryName={a4.secondImageryName}
            onFetchSpiritMatch={fetchSpiritMatch}
            onComplete={goToPostcard}
            completeDelay={5000}
          />
        );
      }

      case STAGES.POSTCARD:
        return (
          <Act5Postcard
            postcardData={buildAct5Data(color, objectResult, matchedCharacter, narrative)}
            autoPlay={true}
            onComplete={() => addLog("明信片生成完成")}
            onRestart={resetAll}
          />
        );

      default:
        return (
          <Act0
            onNext={goToTransition}
            autoAdvanceDelay={4000}
          />
        );
    }
  }

  return (
    <div className="min-h-screen bg-ink text-white">
      {/* Fixed overlay header for act pages */}
      <div className={isActPage ? "fixed top-0 left-0 right-0 z-50 pointer-events-none" : ""}>
        <div className={isActPage ? "pointer-events-auto" : ""}>
          <Header
            mode={mode}
            onModeChange={handleModeChange}
            wsStatus={socket.status}
            wsError={socket.error}
            onConnect={socket.connect}
            onDisconnect={socket.disconnect}
          />
        </div>
      </div>

      {!isActPage && (
        <div className="border-b border-white/5 bg-black/10 px-4 py-4">
          <StageStepper currentStage={currentStage} />
        </div>
      )}

      <main className={isActPage ? "" : "mx-auto max-w-[1280px] px-4 py-6 md:px-8"}>
        {renderCurrentStage()}

        {/* Debug panel — visible during live mode for monitoring */}
        {(mode === "live" || !isActPage) && (
          <details className={mode === "live" && isActPage ? "fixed bottom-4 right-4 z-[100] max-w-sm rounded-xl border border-white/10 bg-black/85 text-sm text-white/60 backdrop-blur" : "mx-auto mt-6 max-w-5xl rounded-xl border border-white/5 bg-black/10 text-sm text-white/45"}>
            <summary className="cursor-pointer px-4 py-3 hover:text-white/65">查看系统状态与事件日志</summary>
            <div className="grid gap-5 border-t border-white/5 p-4 md:grid-cols-2">
              <div>
                <p className="eyebrow mb-3">CURRENT STATE</p>
                <StatusPanel
                  currentStage={currentStage}
                  color={color}
                  colorSource={colorSource}
                  gesture={gesture}
                  objectResult={objectResult}
                  matchedCharacter={matchedCharacter}
                  isAutoAdvancing={isAutoAdvancing}
                />
              </div>
              <div>
                <p className="eyebrow mb-3">SYSTEM LOG</p>
                <SystemLog logs={logs} />
              </div>
            </div>
          </details>
        )}
      </main>
    </div>
  );
}
