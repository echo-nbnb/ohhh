import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_TIMINGS } from "../config/spiritLayout";

export function useSpiritSequence({
  character,
  autoPlay = true,
  timings,
  onComplete,
}) {
  const mergedTimings = useMemo(
    () => ({ ...DEFAULT_TIMINGS, ...(timings ?? {}) }),
    [timings],
  );
  const [status, setStatus] = useState("searching");
  const [visibleSentenceCount, setVisibleSentenceCount] = useState(0);
  const [started, setStarted] = useState(autoPlay);
  const completedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  const characterKey = `${character.id}:${character.name}`;

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const start = useCallback(() => {
    completedRef.current = false;
    setVisibleSentenceCount(0);
    setStatus("searching");
    setStarted(true);
  }, []);

  useEffect(() => {
    completedRef.current = false;
    setStatus("searching");
    setVisibleSentenceCount(0);
    setStarted(autoPlay);
  }, [autoPlay, characterKey]);

  useEffect(() => {
    if (!started) return undefined;

    let timer;
    const schedule = (callback, delay) => {
      timer = window.setTimeout(callback, Math.max(0, Number(delay) || 0));
    };

    if (status === "searching") {
      schedule(() => setStatus("silhouette"), mergedTimings.searchDuration);
    } else if (status === "silhouette") {
      schedule(() => {
        setVisibleSentenceCount(character.monologue.length > 0 ? 1 : 0);
        setStatus("speaking");
      }, mergedTimings.silhouetteDuration);
    } else if (status === "speaking") {
      if (visibleSentenceCount < character.monologue.length) {
        schedule(
          () => setVisibleSentenceCount((count) => count + 1),
          mergedTimings.sentenceDuration,
        );
      } else {
        schedule(() => setStatus("revealing"), mergedTimings.sentenceDuration);
      }
    } else if (status === "revealing") {
      schedule(() => setStatus("revealed"), mergedTimings.revealDuration);
    } else if (status === "revealed" && !completedRef.current) {
      schedule(() => {
        completedRef.current = true;
        onCompleteRef.current?.(character);
      }, mergedTimings.revealedHoldDuration);
    }

    return () => window.clearTimeout(timer);
  }, [
    character,
    mergedTimings,
    started,
    status,
    visibleSentenceCount,
  ]);

  return {
    status,
    visibleSentenceCount,
    started,
    start,
    resolvedTimings: mergedTimings,
  };
}
