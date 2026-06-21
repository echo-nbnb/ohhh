export default function SpiritIdentity({ character }) {
  return (
    <div className="spirit-module-identity">
      <h2 className="spirit-module-name">{character.name}</h2>
      <p className="spirit-module-title">{character.title}</p>
    </div>
  );
}
