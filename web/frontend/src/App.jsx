import { useCallback, useEffect, useRef, useState } from "react";
import Header from "./components/Header";
import StageStepper from "./components/StageStepper";
import StatusPanel from "./components/StatusPanel";
import SystemLog from "./components/SystemLog";
import IntroStage from "./components/stages/IntroStage";
import ColorStage from "./components/stages/ColorStage";
import DrawStage from "./components/stages/DrawStage";
import SpiritStage from "./components/stages/SpiritStage";
import PostcardStage from "./components/stages/PostcardStage";
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

const WS_URL = (() => {
  const { protocol, host } = window.location;
  return `${protocol === "https:" ? "wss" : "ws"}://${host}/ws`;
})();

export default function App() {
  const [mode, setMode] = useState("demo");
  const [currentStage, setCurrentStage] = useState("intro");
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

  const timersRef = useRef([]);
  const latestRef = useRef({ mode, color, objectResult, matchedCharacter });

  useEffect(() => {
    latestRef.current = { mode, color, objectResult, matchedCharacter };
  }, [mode, color, objectResult, matchedCharacter]);

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

  const beginSpiritReveal = useCallback((character = null) => {
    clearTimers();
    const latest = latestRef.current;
    const resolvedCharacter = character
      ?? (latest.mode === "demo" && latest.color && latest.objectResult
        ? mockMatchCharacter(latest.color, latest.objectResult)
        : latest.matchedCharacter);

    setCurrentStage("spirit");
    setMatchedCharacter(resolvedCharacter ?? null);
    setSpiritStatus("searching");
    setIsAutoAdvancing(false);
    addLog("进入第三幕：正在千年文脉中寻找回应");

    if (!resolvedCharacter) {
      addLog("Live Mode 正在等待后端返回人物匹配结果");
      return;
    }

    schedule(() => {
      setSpiritStatus("found");
      addLog("找到了回应者，姓名暂时隐藏");
    }, 1200);

    schedule(() => {
      setSpiritStatus("speaking");
      addLog("人物第一人称回声开始");
    }, 2600);

    schedule(() => {
      setSpiritStatus("revealed");
      addLog(`揭示人物：${resolvedCharacter.name}`);
    }, 6500);
  }, [addLog, clearTimers, schedule]);

  const applyColorResult = useCallback((detectedColor, source, confidence = null) => {
    clearTimers();
    setCurrentStage("color");
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

    schedule(() => {
      setCurrentStage("draw");
      setGesture((current) => latestRef.current.mode === "demo" ? "index_pointing" : current);
      setIsAutoAdvancing(false);
      addLog("自动进入第二幕：筑景");
    }, 1500);
  }, [addLog, clearTimers, schedule]);

  const applyObjectResult = useCallback((recognizedObject) => {
    clearTimers();
    setCurrentStage("draw");
    setObjectResult(recognizedObject);
    setGesture("drawing_complete");
    setIsAutoAdvancing(true);
    addLog(`筑景完成：线条被叙事化为${recognizedObject.name}`);

    schedule(() => {
      beginSpiritReveal();
    }, 1500);
  }, [addLog, beginSpiritReveal, clearTimers, schedule]);

  const handleBackendPayload = useCallback((payload) => {
    const message = normalizeBackendMessage(payload);

    if (!message) {
      addLog("忽略无法识别的后端消息");
      return;
    }

    switch (message.type) {
      case MESSAGE_TYPES.COLOR_DETECTED:
        applyColorResult(
          findColor(message.colorName),
          message.source,
          message.confidence,
        );
        break;

      case MESSAGE_TYPES.GESTURE_STATE:
        setGesture(message.gesture || message.mode);
        break;

      case MESSAGE_TYPES.DRAWING_POINT:
        setCurrentStage("draw");
        setPoints((current) => [...current, { x: message.x, y: message.y }]);
        break;

      case MESSAGE_TYPES.OBJECT_RECOGNIZED:
        applyObjectResult(message);
        break;

      case MESSAGE_TYPES.CHARACTER_MATCHED:
      case MESSAGE_TYPES.CHARACTERS_RECOMMENDED:
        if (message.character) {
          beginSpiritReveal(message.character);
        } else {
          addLog("人物消息中没有可用人物数据");
        }
        break;

      case MESSAGE_TYPES.NARRATIVE_GENERATED:
        clearTimers();
        setNarrative(message);
        setCurrentStage("postcard");
        setIsAutoAdvancing(false);
        addLog("收到 AI 叙事，进入第四幕：成色");
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
    applyObjectResult,
    beginSpiritReveal,
    clearTimers,
  ]);

  const socket = useWebSocket(WS_URL, handleBackendPayload);

  const resetAll = useCallback(() => {
    clearTimers();
    setCurrentStage("intro");
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
  }, [clearTimers]);

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

  const handleStartIntro = () => {
    setCurrentStage("color");
    addLog("开始寻色：进入第一幕");
  };

  const handleMockColorDetection = () => {
    if (isAutoAdvancing) return;
    const result = mockDetectColor();
    applyColorResult(result.color, result.source, result.confidence);
  };

  const handleCompleteDrawing = () => {
    if (isAutoAdvancing) return;
    if (!color || points.length === 0) {
      addLog("请先在画布上留下轨迹");
      return;
    }
    applyObjectResult(mockRecognizeObject(points, color));
  };

  const handleEnterPostcard = () => {
    if (!matchedCharacter || spiritStatus !== "revealed") return;
    clearTimers();
    setNarrative(mockGenerateNarrative({ color, objectResult, matchedCharacter }));
    setCurrentStage("postcard");
    setIsAutoAdvancing(false);
    addLog("生成 Mock 叙事并进入第四幕：成色");
  };

  function renderCurrentStage() {
    switch (currentStage) {
      case "intro":
        return <IntroStage onStart={handleStartIntro} />;

      case "color":
        return (
          <ColorStage
            mode={mode}
            color={color}
            colorSource={colorSource}
            onDetect={handleMockColorDetection}
            isAutoAdvancing={isAutoAdvancing}
          />
        );

      case "draw":
        return (
          <DrawStage
            mode={mode}
            color={color}
            points={points}
            objectResult={objectResult}
            onPointsChange={setPoints}
            onClear={() => {
              setPoints([]);
              setObjectResult(null);
              setGesture(mode === "demo" ? "index_pointing" : gesture);
              addLog("清空轨迹，重新筑景");
            }}
            onComplete={handleCompleteDrawing}
            isAutoAdvancing={isAutoAdvancing}
          />
        );

      case "spirit":
        return (
          <SpiritStage
            status={spiritStatus}
            character={matchedCharacter}
            onEnterPostcard={handleEnterPostcard}
          />
        );

      case "postcard":
        return (
          <PostcardStage
            color={color}
            objectResult={objectResult}
            character={matchedCharacter}
            narrative={narrative}
            onRestart={resetAll}
          />
        );

      default:
        return <IntroStage onStart={handleStartIntro} />;
    }
  }

  return (
    <div className="min-h-screen bg-ink text-white">
      <Header
        mode={mode}
        onModeChange={handleModeChange}
        wsStatus={socket.status}
        wsError={socket.error}
        onConnect={socket.connect}
        onDisconnect={socket.disconnect}
      />

      <div className="border-b border-white/5 bg-black/10 px-4 py-4">
        <StageStepper currentStage={currentStage} />
      </div>

      <main className="mx-auto max-w-[1280px] px-4 py-6 md:px-8">
        {renderCurrentStage()}

        <details className="mx-auto mt-6 max-w-5xl rounded-xl border border-white/5 bg-black/10 text-sm text-white/45">
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
      </main>
    </div>
  );
}
