export default function SceneNarration({ scene, color, objectResult }) {
  return (
    <section className="panel">
      <p className="eyebrow">CURRENT NARRATIVE</p>
      <h2 className="mt-2 text-xl font-medium text-paper">{scene.title}</h2>
      <div className="mt-5 grid gap-4 text-sm leading-7 text-white/65 md:grid-cols-2">
        <div>
          <p className="label">投影画面</p>
          <p>{scene.projection}</p>
        </div>
        <div>
          <p className="label">AI 旁白</p>
          <p>{scene.ai}</p>
        </div>
        <div>
          <p className="label">用户提示</p>
          <p>{scene.prompt}</p>
        </div>
        <div>
          <p className="label">过渡文案</p>
          <p>{scene.transition}</p>
        </div>
      </div>
      {color && (
        <div className="mt-5 rounded-lg border border-gold/25 bg-gold/5 px-4 py-3 text-sm text-paper/80">
          “{color.voiceLine}”
          {objectResult && <span className="mt-1 block text-white/55">{objectResult.narrative}</span>}
        </div>
      )}
    </section>
  );
}
