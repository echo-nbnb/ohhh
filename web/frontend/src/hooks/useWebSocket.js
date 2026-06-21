import { useCallback, useEffect, useRef, useState } from "react";

export function useWebSocket(url, onMessage) {
  const socketRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const [status, setStatus] = useState("disconnected");
  const [error, setError] = useState("");

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    setStatus("disconnected");
  }, []);

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;
    setError("");
    setStatus("connecting");
    console.log("[WS] Connecting to:", url);

    try {
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        console.log("[WS] Connected successfully");
        setStatus("connected");
      };
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("[WS] ← Raw message:", data.type, data);
          onMessageRef.current?.(data);
        } catch (e) {
          console.error("[WS] Parse error:", e, "raw:", event.data?.slice?.(0, 200));
          setError("收到无法解析的 WebSocket 消息。");
        }
      };
      socket.onerror = (e) => {
        console.error("[WS] Error event:", e);
        setError("无法连接后端，请确认 Python 服务已启动。");
        setStatus("error");
      };
      socket.onclose = (e) => {
        console.log("[WS] Closed — code:", e.code, "reason:", e.reason);
        socketRef.current = null;
        setStatus((current) => (current === "error" ? current : "disconnected"));
      };
    } catch (e) {
      console.error("[WS] Connection exception:", e);
      setError("WebSocket 地址无效或浏览器拒绝连接。");
      setStatus("error");
    }
  }, [url]);

  // Log status changes
  useEffect(() => {
    console.log("[WS] Status changed to:", status, status === "error" ? `(error: ${error})` : "");
  }, [status, error]);

  useEffect(() => () => socketRef.current?.close(), []);

  return { status, error, connect, disconnect };
}
