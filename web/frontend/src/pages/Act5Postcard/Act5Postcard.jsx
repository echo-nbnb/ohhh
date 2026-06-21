import { useEffect, useMemo, useRef, useState } from "react";
import "./Act5Postcard.css";

import DualColorLiquidChrome from "../../components/DualColorLiquidChrome/DualColorLiquidChrome";

import bgUrl from "../../assets/act5/act5-bg.svg";
import colorRing1Url from "../../assets/act5/color-ring-1.svg";
import colorRing2Url from "../../assets/act5/color-ring-2.svg";
import colorDiskUrl from "../../assets/act5/color-disk.svg";
import colorDiskOrbitUrl from "../../assets/act5/color-disk-orbit.svg";
import personDiskUrl from "../../assets/act5/person-disk.svg";
import titleBgUrl from "../../assets/act5/title-bg.svg";
import pattern1Url from "../../assets/act5/pattern-1.svg";
import pattern2Url from "../../assets/act5/pattern-2.svg";
import mainTitleUrl from "../../assets/act5/main-title.svg";
import mockBridgeUrl from "../../assets/act5/mock-bridge.svg";
import mockTreeUrl from "../../assets/act5/mock-tree.svg";
import mockPersonUrl from "../../assets/act5/mock-person.svg";
import iconUrl from "../../assets/act0/act0-icon.svg";

function random(min, max) { return Math.random() * (max - min) + min; }
function createFloatingIcons(count = 6) {
  const areas = [{ xMin: -8, xMax: 15, yMin: 5, yMax: 95 }, { xMin: 85, xMax: 108, yMin: 5, yMax: 95 }, { xMin: -8, xMax: 15, yMin: 5, yMax: 95 }, { xMin: 85, xMax: 108, yMin: 5, yMax: 95 }];
  return Array.from({ length: count }).map((_, i) => { const a = areas[i % areas.length]; return { id: i, x: random(a.xMin, a.xMax), y: random(a.yMin, a.yMax), size: random(36, 110), opacity: random(0.25, 0.60), rotate: random(0, 360), duration: random(10, 22), delay: random(-18, 0), driftX: random(-90, 90), driftY: random(-70, 70) }; });
}

const STAGE = { MAKING: 0, PLACES_AND_BRIDGE: 1, TREE_APPEAR: 2, DISKS_APPEAR: 3, TITLE_BG_APPEAR: 4, TOP_TITLE_APPEAR: 5, OBJECT_TEXT_APPEAR: 6, AI_TEXT_APPEAR: 7, TRACE_APPEAR: 8, SEAL_READY: 9, FINAL_POSTCARD: 10 };

function createMaskStyle(url, bg, op = 1) {
  return { background: bg, opacity: op, WebkitMaskImage: `url(${url})`, maskImage: `url(${url})`, WebkitMaskRepeat: "no-repeat", maskRepeat: "no-repeat", WebkitMaskSize: "100% 100%", maskSize: "100% 100%", WebkitMaskPosition: "center", maskPosition: "center" };
}

function getNowText() {
  const now = new Date();
  return `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, "0")}.${String(now.getDate()).padStart(2, "0")}\n${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
}

const MOCK_DATA = {
  colors: { primary: "#F2E700", secondary: "#2F55F6", primaryName: "桂黄", secondaryName: "澄蓝" },
  selectedPlaces: ["岳麓山", "湘江水", "书院檐角"],
  title: { cn: "桂黄映桥，澄蓝问道", en: "BETWEEN OSMANTHUS YELLOW AND CLEAR BLUE,\nZHANG SHI STILL SEEKS THE WAY." },
  traceText: "[桂黄·澄蓝] → [桥｜树] → [张栻]",
  imageryItems: [{ id: "bridge", name: "桥", imageUrl: mockBridgeUrl, className: "act5__imagery--bridge" }, { id: "tree", name: "树", imageUrl: mockTreeUrl, className: "act5__imagery--tree" }],
  objectText: ["蓝与桥相遇，这是通往远方的路。", "黄与古树相遇，这是根"],
  aiWriting: ["刚才与你说话的，是我。", "我曾在树影下讲学，", "也曾在桥畔望向来路。", "桥连接的不只是此岸与彼岸，", "也是今日的你，", "与千年前仍未停息的追问。", "愿你在桂黄的光里，", "保有澄蓝的心；", "敢问，敢辨，", "也敢向更远处走去。"],
  person: { name: "张栻", portraitUrl: mockPersonUrl },
  mainTitleImageUrl: mainTitleUrl, downloadQrUrl: "", createdAtText: getNowText(),
};

export default function Act5Postcard({ postcardData = MOCK_DATA, debugStage, autoPlay = true, onFetchPostcardData, onComplete, onRestart, completeDelay = 5000 }) {
  const [stage, setStage] = useState(debugStage ?? STAGE.MAKING);
  const [data, setData] = useState(postcardData);
  const floatingIcons = useMemo(() => createFloatingIcons(6), []);
  const onCompleteRef = useRef(onComplete);
  const completeDelayRef = useRef(completeDelay);
  useEffect(() => { onCompleteRef.current = onComplete; }, [onComplete]);
  useEffect(() => { completeDelayRef.current = completeDelay; }, [completeDelay]);

  useEffect(() => { if (debugStage !== undefined) setStage(debugStage); }, [debugStage]);
  useEffect(() => { let m = true; if (onFetchPostcardData) onFetchPostcardData().then(r => { if (m && r) setData(p => ({ ...p, ...r })); }); return () => { m = false; }; }, [onFetchPostcardData]);

  useEffect(() => {
    if (!autoPlay || debugStage !== undefined) return;
    const ts = [setTimeout(() => setStage(STAGE.PLACES_AND_BRIDGE), 3600), setTimeout(() => setStage(STAGE.TREE_APPEAR), 7300), setTimeout(() => setStage(STAGE.DISKS_APPEAR), 10900), setTimeout(() => setStage(STAGE.TITLE_BG_APPEAR), 14600), setTimeout(() => setStage(STAGE.TOP_TITLE_APPEAR), 18200), setTimeout(() => setStage(STAGE.OBJECT_TEXT_APPEAR), 22100), setTimeout(() => setStage(STAGE.AI_TEXT_APPEAR), 26000), setTimeout(() => setStage(STAGE.TRACE_APPEAR), 30000), setTimeout(() => setStage(STAGE.SEAL_READY), 33900), setTimeout(() => { setStage(STAGE.FINAL_POSTCARD); if (onCompleteRef.current) setTimeout(() => onCompleteRef.current(), completeDelayRef.current); }, 38100)];
    return () => ts.forEach(clearTimeout);
  }, [autoPlay, debugStage]);

  const primaryColor = data.colors?.primary || "#F2E700";
  const secondaryColor = data.colors?.secondary || "#2F55F6";

  const centerLines = useMemo(() => {
    if (stage === STAGE.MAKING) return ["你的千年色正在成笺……"];
    if (stage === STAGE.PLACES_AND_BRIDGE || stage === STAGE.TREE_APPEAR) { const p = data.selectedPlaces?.join("、") || "岳麓山、湘江水、书院檐角"; return [`${p}`, "角……正在浮现"]; }
    if (stage === STAGE.DISKS_APPEAR || stage === STAGE.TITLE_BG_APPEAR) return ["你的千年色正在成笺……"];
    if (stage >= STAGE.TOP_TITLE_APPEAR && stage <= STAGE.TRACE_APPEAR) return ["墨色正在扩散", "字句正在成形……"];
    if (stage === STAGE.SEAL_READY) return ["握拳盖章"];
    return [];
  }, [stage, data.selectedPlaces]);

  const showColorDecor = stage >= STAGE.PLACES_AND_BRIDGE, showBridge = stage >= STAGE.PLACES_AND_BRIDGE, showTree = stage >= STAGE.TREE_APPEAR, showDisks = stage >= STAGE.DISKS_APPEAR, showTitleBg = stage >= STAGE.TITLE_BG_APPEAR, showTopTitle = stage >= STAGE.TOP_TITLE_APPEAR, showObjectText = stage >= STAGE.OBJECT_TEXT_APPEAR, showAiText = stage >= STAGE.AI_TEXT_APPEAR, showTrace = stage >= STAGE.TRACE_APPEAR, showSeal = stage >= STAGE.SEAL_READY, showFinal = stage >= STAGE.FINAL_POSTCARD;

  return (
    <section className="act5">
      <img className="act5__bg" src={bgUrl} alt="" draggable="false" />
      {showColorDecor && (<><div className="act5__pattern act5__pattern--one" style={createMaskStyle(pattern1Url, primaryColor, 0.95)} /><div className="act5__pattern act5__pattern--two" style={createMaskStyle(pattern2Url, secondaryColor, 0.95)} /></>)}
      {showColorDecor && (<><div className="act5__colorRing act5__colorRing--one" style={createMaskStyle(colorRing1Url, primaryColor, 1)} /><div className="act5__colorRing act5__colorRing--two" style={createMaskStyle(colorRing2Url, secondaryColor, 1)} /></>)}
      {showTitleBg && <img className="act5__titleBg" src={titleBgUrl} alt="" draggable="false" />}
      {showTopTitle && (<div className="act5__postcardTitle"><div className="act5__postcardTitleCn">{data.title?.cn}</div><div className="act5__postcardTitleEn">{(data.title?.en || "").split("\n").map((l, i) => <div key={`te-${i}`}>{l}</div>)}</div></div>)}
      {showTrace && <div className="act5__traceText">{data.traceText}</div>}
      {showBridge && <img className="act5__imagery act5__imagery--bridge" src={data.imageryItems?.[0]?.imageUrl || mockBridgeUrl} alt="桥" draggable="false" />}
      {showTree && <img className="act5__imagery act5__imagery--tree" src={data.imageryItems?.[1]?.imageUrl || mockTreeUrl} alt="树" draggable="false" />}
      {showDisks && (<><div className="act5__colorDiskWrap"><div className="act5__liquidDisk"><DualColorLiquidChrome colorA={primaryColor} colorB={secondaryColor} /></div><img className="act5__colorDiskFrame" src={colorDiskUrl} alt="" draggable="false" /><img className="act5__colorDiskOrbit" src={colorDiskOrbitUrl} alt="" draggable="false" /></div><div className="act5__personDiskWrap"><div className="act5__personInner"><img className="act5__personPortrait" src={data.person?.portraitUrl || mockPersonUrl} alt={data.person?.name || "人物"} draggable="false" /></div><img className="act5__personDisk" src={personDiskUrl} alt="" draggable="false" /></div></>)}
      {showObjectText && (<div className="act5__objectText">{(data.objectText || []).map((l, i) => <div key={`ot-${i}`}>{l}</div>)}</div>)}
      {showAiText && (<div className="act5__aiWriting">{(data.aiWriting || []).map((l, i) => <div key={`aw-${i}`}>{l}</div>)}</div>)}
      {centerLines.length > 0 && !showFinal && (<div className="act5__centerCopy" key={`c-${stage}`}>{centerLines.map((l, i) => <div className="act5__centerLine" key={`${l}-${i}`}>{l}</div>)}</div>)}
      {showSeal && (<div className="act5__dateText">{(data.createdAtText || getNowText()).split("\n").map((l, i) => <div key={`d-${i}`}>{l}</div>)}</div>)}
      {showFinal && <img className="act5__mainTitle" src={data.mainTitleImageUrl || mainTitleUrl} alt="寻麓千年色" draggable="false" />}
      {showFinal && (<div className="act5__qrArea">{data.downloadQrUrl ? <img className="act5__qrImage" src={data.downloadQrUrl} alt="QR" draggable="false" /> : <div className="act5__qrPlaceholder"><span>QR</span></div>}</div>)}
      {showFinal && onRestart && (
        <button className="act5__restartBtn" type="button" onClick={onRestart}>重新开始</button>
      )}
      <div className="act5__icons" aria-hidden="true">
        {floatingIcons.map(item => (<img key={item.id} className="act5__icon" src={iconUrl} alt="" draggable="false" style={{ left: `${item.x}%`, top: `${item.y}%`, width: `${item.size}px`, height: `${item.size}px`, opacity: item.opacity, transform: `rotate(${item.rotate}deg)`, "--duration": `${item.duration}s`, "--delay": `${item.delay}s`, "--drift-x": `${item.driftX}px`, "--drift-y": `${item.driftY}px` }} />))}
      </div>
    </section>
  );
}
