import { useEffect, useMemo, useState } from "react";
import fallbackSilhouette from "../assets/default-silhouette.svg";

export default function CharacterReveal({
  character,
  status,
  resolveCharacterImage,
}) {
  const preferredSource = useMemo(() => {
    if (character.image) return character.image;
    if (typeof resolveCharacterImage === "function") {
      return resolveCharacterImage(character) || fallbackSilhouette;
    }
    return fallbackSilhouette;
  }, [character, resolveCharacterImage]);
  const [source, setSource] = useState(preferredSource);

  useEffect(() => setSource(preferredSource), [preferredSource]);

  const isClear = status === "revealing" || status === "revealed";

  return (
    <img
      className={`spirit-module-character ${
        isClear ? "spirit-module-character--clear" : "spirit-module-character--hidden"
      }`}
      src={source}
      alt={status === "revealed" ? character.name : "正在显影的人物"}
      onError={() => {
        if (source !== fallbackSilhouette) setSource(fallbackSilhouette);
      }}
    />
  );
}
