const statusLabels = {
  disconnected: "未连接",
  connecting: "连接中",
  connected: "已连接",
  error: "连接失败",
};

export default function Header({
  mode,
  onModeChange,
  wsStatus,
  wsError,
  onConnect,
  onDisconnect,
}) {
  return (
    <header className="border-b border-white/10 bg-black/20 px-5 py-4">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-gold">AI Cultural Interaction System</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-[0.18em] text-paper">寻麓千年色</h1>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-sm">
          <div className="flex rounded-lg border border-white/10 bg-black/20 p-1">
            {["demo", "live"].map((item) => (
              <button
                key={item}
                className={`rounded-md px-3 py-2 ${mode === item ? "bg-gold text-ink" : "text-white/60 hover:text-white"}`}
                onClick={() => onModeChange(item)}
              >
                {item === "demo" ? "Demo Mode" : "Live Mode"}
              </button>
            ))}
          </div>

          <span className={`status-dot status-${wsStatus}`} />
          <span className="text-white/70">WebSocket：{statusLabels[wsStatus]}</span>
          {mode === "live" && (
            <button
              className="button-secondary"
              onClick={wsStatus === "connected" ? onDisconnect : onConnect}
              disabled={wsStatus === "connecting"}
            >
              {wsStatus === "connected" ? "断开后端" : "连接后端"}
            </button>
          )}
        </div>
      </div>
      {mode === "live" && wsError && (
        <p className="mx-auto mt-3 max-w-[1600px] rounded-md bg-red-950/50 px-3 py-2 text-sm text-red-200">{wsError}</p>
      )}
    </header>
  );
}
