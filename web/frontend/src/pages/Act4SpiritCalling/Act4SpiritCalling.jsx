import { useEffect, useMemo, useRef, useState } from "react";
import "./Act4SpiritCalling.css";

import bgUrl from "../../assets/act4/act4-bg.svg";
import titleTopUrl from "../../assets/act4/title-top.svg";
import titleBottomUrl from "../../assets/act4/title-bottom.svg";
import barTopUrl from "../../assets/act4/bar-top.svg";
import barBottomUrl from "../../assets/act4/bar-bottom.svg";
import personFrameUrl from "../../assets/act4/person-frame.svg";
import personMockUrl from "../../assets/act4/person-zhangshi.svg";
import blocksPrimaryUrl from "../../assets/act4/blocks-primary.svg";
import blocksSecondaryUrl from "../../assets/act4/blocks-secondary.svg";
import iconUrl from "../../assets/act0/act0-icon.svg";

function random(min, max) { return Math.random() * (max - min) + min; }
function createFloatingIcons(count = 6) {
  const areas = [
    { xMin: -8, xMax: 15, yMin: 5, yMax: 95 },
    { xMin: 85, xMax: 108, yMin: 5, yMax: 95 },
    { xMin: -8, xMax: 15, yMin: 5, yMax: 95 },
    { xMin: 85, xMax: 108, yMin: 5, yMax: 95 },
  ];
  return Array.from({ length: count }).map((_, i) => {
    const a = areas[i % areas.length];
    return { id: i, x: random(a.xMin, a.xMax), y: random(a.yMin, a.yMax), size: random(36, 110), opacity: random(0.25, 0.60), rotate: random(0, 360), duration: random(10, 22), delay: random(-18, 0), driftX: random(-90, 90), driftY: random(-70, 70) };
  });
}

function createMaskStyle(url, bg, op = 1) {
  return { background: bg, opacity: op, WebkitMaskImage: `url(${url})`, maskImage: `url(${url})`, WebkitMaskRepeat: "no-repeat", maskRepeat: "no-repeat", WebkitMaskSize: "100% 100%", maskSize: "100% 100%", WebkitMaskPosition: "center", maskPosition: "center" };
}

async function mockFetchSpiritMatch() {
  await new Promise((r) => setTimeout(r, 1200));
  return {
    person: { id: "zhang-shi", name: "张栻", subtitle: ["岳麓书院早期", "讲学者之一。"], portraitUrl: personMockUrl },
    narrative: {
      centerStart: ["颜色已经展开", "意象也已经落下。"],
      centerSeek: ["现在", "我要在千年的文脉里", "寻找一个与你相遇的人。"],
      loading: ["正在寻找回应你的人……"],
      found: ["找到了！"],
      rightInterim: ["他还不能告诉你名字", "你要先听他说完。"],
      leftBlue: ["你选择了蓝。", "又画下桥。", "我知道那种颜色。", "那不是遥远的蓝，", "是一个人走向彼岸时，", "仍然愿意回头发问的蓝。", "后来者，", "你是想寻找答案，", "还是想成为那个继续提问的人？"],
      leftYellow: ["你选择了黄。", "又画下树。", "我知道那种颜色。", "那不是喧闹的黄，", "是一个人站在树影之下，", "仍然愿意向光发问的黄。", "后来者，", "你是想寻找答案，", "还是想成为那个继续提问的人？"],
      rightFinal: ["刚才与你说话的，是他。", "", "但他留下的不", "只是名字", "", "更是一种敢于", "发问的底色。"],
    },
  };
}

const STAGE = { INTRO_1: 0, INTRO_2: 1, LOADING: 2, FOUND: 3, FRAME_EMPTY: 4, BLUE_TEXT: 5, YELLOW_TEXT: 6, FINAL_REVEAL: 7 };

export default function Act4SpiritCalling({
  primaryColor = "#F2E700", secondaryColor = "#355BFF",
  firstColorName = "蓝", secondColorName = "黄",
  firstImageryName = "桥", secondImageryName = "树",
  onFetchSpiritMatch,
  onComplete,
  completeDelay = 4000,
  waitingForStamp = false,
}) {
  const [stage, setStage] = useState(STAGE.INTRO_1);
  const [matchData, setMatchData] = useState(null);
  const floatingIcons = useMemo(() => createFloatingIcons(6), []);
  const onCompleteRef = useRef(onComplete);
  const completeDelayRef = useRef(completeDelay);
  useEffect(() => { onCompleteRef.current = onComplete; }, [onComplete]);
  useEffect(() => { completeDelayRef.current = completeDelay; }, [completeDelay]);

  useEffect(() => {
    let mounted = true;
    const payload = { scene: "act4-spirit-calling", colors: { primary: primaryColor, secondary: secondaryColor, firstColorName, secondColorName }, imagery: { first: firstImageryName, second: secondImageryName } };
    (async () => { try { const r = onFetchSpiritMatch ? await onFetchSpiritMatch(payload) : await mockFetchSpiritMatch(); if (mounted) setMatchData(r); } catch (e) { console.error(e); } })();
    const ts = [];
    ts.push(setTimeout(() => setStage(STAGE.INTRO_2), 3600));
    ts.push(setTimeout(() => setStage(STAGE.LOADING), 8000));
    ts.push(setTimeout(() => setStage(STAGE.FOUND), 12000));
    ts.push(setTimeout(() => setStage(STAGE.FRAME_EMPTY), 13700));
    ts.push(setTimeout(() => setStage(STAGE.BLUE_TEXT), 16200));
    ts.push(setTimeout(() => setStage(STAGE.YELLOW_TEXT), 23800));
    ts.push(setTimeout(() => { setStage(STAGE.FINAL_REVEAL); if (onCompleteRef.current) setTimeout(() => onCompleteRef.current(), completeDelayRef.current); }, 31900));
    return () => { mounted = false; ts.forEach(clearTimeout); };
  }, [primaryColor, secondaryColor, firstColorName, secondColorName, firstImageryName, secondImageryName, onFetchSpiritMatch]);

  const narrative = matchData?.narrative;
  const person = matchData?.person;

  const centerLines = useMemo(() => {
    if (!narrative) { if (stage === STAGE.INTRO_1) return ["颜色已经展开", "意象也已经落下。"]; if (stage === STAGE.INTRO_2) return ["现在", "我要在千年的文脉里", "寻找一个与你相遇的人。"]; if (stage === STAGE.LOADING) return ["正在寻找回应你的人……"]; if (stage === STAGE.FOUND) return ["找到了！"]; return []; }
    if (stage === STAGE.INTRO_1) return narrative.centerStart || [];
    if (stage === STAGE.INTRO_2) return narrative.centerSeek || [];
    if (stage === STAGE.LOADING) return narrative.loading || [];
    if (stage === STAGE.FOUND) return narrative.found || [];
    return [];
  }, [stage, narrative]);

  const leftLines = useMemo(() => {
    if (!narrative || !person) return [];
    if (stage === STAGE.BLUE_TEXT) return narrative.leftBlue || [];
    if (stage === STAGE.YELLOW_TEXT) return narrative.leftYellow || [];
    if (stage === STAGE.FINAL_REVEAL) return [person.name, ...(person.subtitle || [])];
    return [];
  }, [stage, narrative, person]);

  const rightLines = useMemo(() => {
    if (!narrative) return [];
    if (stage === STAGE.FRAME_EMPTY || stage === STAGE.BLUE_TEXT || stage === STAGE.YELLOW_TEXT) return narrative.rightInterim || [];
    if (stage === STAGE.FINAL_REVEAL) return narrative.rightFinal || [];
    return [];
  }, [stage, narrative]);

  const showFrame = stage >= STAGE.FRAME_EMPTY;
  const showPortrait = stage >= STAGE.FINAL_REVEAL;
  const showBlocks = stage >= STAGE.FINAL_REVEAL;

  return (
    <section className="act4">
      <img className="act4__bg" src={bgUrl} alt="" draggable="false" />
      <div className="act4__bar act4__bar--top" style={createMaskStyle(barTopUrl, primaryColor, 1)} />
      <div className="act4__bar act4__bar--bottom" style={createMaskStyle(barBottomUrl, secondaryColor, 1)} />
      <div className="act4__centerStage">
        {showFrame && <img className="act4__personFrame" src={personFrameUrl} alt="" draggable="false" />}
        {showPortrait && <img className="act4__personPortrait" src={(person?.portraitUrl && person.portraitUrl !== "") ? person.portraitUrl : personMockUrl} alt={person?.name || "人物"} draggable="false" />}
        {showBlocks && (<><div className="act4__blocks act4__blocks--primary" style={createMaskStyle(blocksPrimaryUrl, primaryColor, 1)} /><div className="act4__blocks act4__blocks--secondary" style={createMaskStyle(blocksSecondaryUrl, secondaryColor, 1)} /></>)}
      </div>
      {centerLines.length > 0 && (<div className="act4__centerCopy" key={`c-${stage}`}>{centerLines.map((l, i) => <div className="act4__centerLine" key={`${l}-${i}`}>{l}</div>)}</div>)}
      {leftLines.length > 0 && (<div className={["act4__sideCopy", "act4__sideCopy--left", stage === STAGE.FINAL_REVEAL ? "is-final" : ""].join(" ")} key={`l-${stage}`}>{leftLines.map((l, i) => <div className="act4__sideLine" key={`${l}-${i}`}>{l}</div>)}</div>)}
      {rightLines.length > 0 && (<div className={["act4__sideCopy", "act4__sideCopy--right", stage === STAGE.FINAL_REVEAL ? "is-final" : ""].join(" ")} key={`r-${stage}`}>{rightLines.map((l, i) => <div className="act4__sideLine" key={`${l}-${i}`}>{l || " "}</div>)}</div>)}
      <img className="act4__title act4__title--top" src={titleTopUrl} alt="" draggable="false" />
      <img className="act4__title act4__title--bottom" src={titleBottomUrl} alt="" draggable="false" />
      <div className="act4__icons" aria-hidden="true">
        {floatingIcons.map((item) => (<img key={item.id} className="act4__icon" src={iconUrl} alt="" draggable="false" style={{ left: `${item.x}%`, top: `${item.y}%`, width: `${item.size}px`, height: `${item.size}px`, opacity: item.opacity, transform: `rotate(${item.rotate}deg)`, "--duration": `${item.duration}s`, "--delay": `${item.delay}s`, "--drift-x": `${item.driftX}px`, "--drift-y": `${item.driftY}px` }} />))}
      </div>
      {waitingForStamp && (
        <div className="act4__stampPrompt">
          <div className="act4__stampText">握拳盖章</div>
          <div className="act4__stampSub">完成你的千年色</div>
        </div>
      )}
    </section>
  );
}
