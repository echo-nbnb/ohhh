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

    try {
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => setStatus("connected");
      socket.onmessage = (event) => {
        try {
          onMessageRef.current?.(JSON.parse(event.data));
        } catch {
          setError("收到无法解析的 WebSocket 消息。");
        }
      };
      socket.onerror = () => {
        setError("无法连接后端，请确认 Python 服务已启动。");
        setStatus("error");
      };
      socket.onclose = () => {
        socketRef.current = null;
        setStatus((current) => (current === "error" ? current : "disconnected"));
      };
    } catch {
      setError("WebSocket 地址无效或浏览器拒绝连接。");
      setStatus("error");
    }
  }, [url]);

  useEffect(() => () => socketRef.current?.close(), []);

  return { status, error, connect, disconnect };
}
