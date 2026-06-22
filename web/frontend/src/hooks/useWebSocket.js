import { useCallback, useEffect, useRef, useState } from "react";

export function useWebSocket(url, onMessage) {
  const socketRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const [status, setStatus] = useState("disconnected");
  const [error, setError] = useState("");
  const mountedRef = useRef(true);
  const reconnectTimerRef = useRef(null);
  const intentionalCloseRef = useRef(false);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  // Track mount state for async safety
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      // Clear any pending reconnect timer
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };
  }, []);

  const send = useCallback((data) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(data));
    }
  }, []);

  const disconnect = useCallback(() => {
    intentionalCloseRef.current = true;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (socketRef.current) {
      socketRef.current.close(1000, "intentional");
      socketRef.current = null;
    }
    setStatus("disconnected");
    setError("");
  }, []);

  const connect = useCallback(() => {
    // Don't connect if already connected or connecting
    if (socketRef.current?.readyState === WebSocket.OPEN ||
        socketRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Clean up old socket
    if (socketRef.current) {
      socketRef.current.onclose = null; // prevent auto-reconnect from stale socket
      try { socketRef.current.close(); } catch (e) { /* ignore */ }
      socketRef.current = null;
    }

    // Clear stale reconnect timer
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    intentionalCloseRef.current = false;
    setError("");
    setStatus("connecting");
    console.log("[WS] Connecting to:", url);

    try {
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        if (!mountedRef.current) return;
        console.log("[WS] Connected successfully");
        setStatus("connected");
        setError("");
      };

      socket.onmessage = (event) => {
        if (!mountedRef.current) return;
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
        if (!mountedRef.current) return;
        console.error("[WS] Error event:", e);
        setError("无法连接后端，请确认 Python 服务已启动。");
        setStatus("error");
      };

      socket.onclose = (e) => {
        if (!mountedRef.current) return;
        console.log("[WS] Closed — code:", e.code, "reason:", e.reason);
        socketRef.current = null;
        setStatus((current) => (current === "error" ? current : "disconnected"));

        // Only auto-reconnect if not intentional and still mounted
        if (!intentionalCloseRef.current && e.code !== 1000 && mountedRef.current) {
          console.log("[WS] Scheduling auto-reconnect in 2s");
          reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null;
            if (mountedRef.current) {
              connect();
            }
          }, 2000);
        }
      };
    } catch (e) {
      console.error("[WS] Connection exception:", e);
      if (mountedRef.current) {
        setError("WebSocket 地址无效或浏览器拒绝连接。");
        setStatus("error");
      }
    }
  }, [url]);

  // Log status changes
  useEffect(() => {
    console.log("[WS] Status changed to:", status, status === "error" ? `(error: ${error})` : "");
  }, [status, error]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      intentionalCloseRef.current = true;
      if (socketRef.current) {
        socketRef.current.onclose = null;
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, []);

  return { status, error, connect, disconnect, send };
}
