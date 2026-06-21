function normalizeLines(value) {
  if (Array.isArray(value)) {
    return value.map((line) => String(line).trim()).filter(Boolean);
  }
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

export function normalizeCharacter(character) {
  const source = character && typeof character === "object" ? character : {};

  return {
    ...source,
    id: source.id == null ? "" : String(source.id),
    name: source.name == null || source.name === "" ? "无名回应者" : String(source.name),
    title: source.title == null ? "" : String(source.title),
    image: typeof source.image === "string" ? source.image : "",
    monologue: normalizeLines(source.monologue),
    spiritLine: normalizeLines(source.spiritLine),
  };
}
