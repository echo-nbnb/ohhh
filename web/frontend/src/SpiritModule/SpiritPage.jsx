import { useMemo } from "react";
import CharacterReveal from "./components/CharacterReveal";
import SpiritCanvas from "./components/SpiritCanvas";
import SpiritIdentity from "./components/SpiritIdentity";
import SpiritMonologue from "./components/SpiritMonologue";
import { regionStyle, SPIRIT_LAYOUT } from "./config/spiritLayout";
import { useSpiritSequence } from "./hooks/useSpiritSequence";
import { normalizeCharacter } from "./utils/normalizeCharacter";
import "./SpiritPage.css";

export default function SpiritPage({
  character,
  onComplete,
  autoPlay = true,
  timings,
  resolveCharacterImage,
  className = "",
  style,
}) {
  const normalizedCharacter = useMemo(
    () => normalizeCharacter(character),
    [character],
  );
  const {
    status,
    visibleSentenceCount,
    started,
    start,
    resolvedTimings,
  } = useSpiritSequence({
    character: normalizedCharacter,
    autoPlay,
    timings,
    onComplete,
  });
  const revealed = status === "revealed";
  const showUnknown = ["silhouette", "speaking", "revealing"].includes(status);
  const showMonologue = status === "speaking" || status === "revealing";

  return (
    <SpiritCanvas
      className={className}
      style={{
        ...style,
        "--spirit-module-reveal-duration": `${resolvedTimings.revealDuration}ms`,
      }}
    >
      <div
        className="spirit-module-character-region"
        style={regionStyle(SPIRIT_LAYOUT.portrait)}
      >
        <CharacterReveal
          character={normalizedCharacter}
          status={status}
          resolveCharacterImage={resolveCharacterImage}
        />
      </div>

      {showMonologue && (
        <div style={regionStyle(SPIRIT_LAYOUT.monologue)}>
          <SpiritMonologue
            lines={normalizedCharacter.monologue}
            visibleCount={visibleSentenceCount}
          />
        </div>
      )}

      {showUnknown && (
        <div
          className="spirit-module-unknown"
          style={regionStyle(SPIRIT_LAYOUT.unknownName)}
        >
          ????
        </div>
      )}

      {revealed && (
        <>
          <div style={regionStyle(SPIRIT_LAYOUT.identity)}>
            <SpiritIdentity character={normalizedCharacter} />
          </div>
          <div
            className="spirit-module-spirit-lines"
            style={regionStyle(SPIRIT_LAYOUT.message)}
          >
            {normalizedCharacter.spiritLine.map((line, index) => (
              <p key={`${index}-${line}`}>{line}</p>
            ))}
          </div>
        </>
      )}

      {!revealed && status !== "speaking" && status !== "revealing" && (
        <div
          className="spirit-module-search-message"
          style={regionStyle(SPIRIT_LAYOUT.message)}
        >
          <p>{status === "searching" ? "正在寻找" : "灵息已经靠近"}</p>
          <p>{status === "searching" ? "回应你的人……" : "请听他说完"}</p>
        </div>
      )}

      {!started && (
        <div style={regionStyle(SPIRIT_LAYOUT.startButton)}>
          <button
            type="button"
            className="spirit-module-start-button"
            onClick={start}
          >
            开始唤灵
          </button>
        </div>
      )}

      <div
        className="spirit-module-status"
        style={regionStyle(SPIRIT_LAYOUT.status)}
      >
        {status}
      </div>
    </SpiritCanvas>
  );
}
