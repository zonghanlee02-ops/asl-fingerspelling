// Live reading loop over a WebSocket with final_model_i: /ws/live (TCN + CTC + live word
// decoder), 30 fps, each frame sent with the time it was captured, so the model sees the real
// signing speed.
// With ?replay=1 in the page address (and ASL_LIVE_REPLAY set on the server) the server plays a
// saved take instead of the camera, for checks and screenshots without a webcam.
//
// One frame is "in flight" at a time: the next camera frame is captured
// only after the backend has replied to the previous one. So the backend
// always works on the newest frame and no queue of old frames builds up.
import { useCallback, useEffect, useRef, useState } from "react";
import { NO_BACKEND, wsUrl } from "./api.js";

const MAX_FPS = 30;                // final_model_i was trained at 30 frames per second
const SEND_WIDTH = 640;           // your recordings are 640 x 480
export const REPLAY = new URLSearchParams(location.search).has("replay");

export function useRecognition({ videoRef, active, onResult }) {
  const wsRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const inFlight = useRef(false);
  const lastSent = useRef(0);
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;
  const [status, setStatus] = useState("idle");    // idle | connecting | open | error
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ fps: 0, latency: 0, process: 0 });
  const timesRef = useRef([]);

  const send = useCallback((msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(msg)); return true; }
    return false;
  }, []);

  // Open the socket while reading. If the backend cannot be reached (still starting, restarted),
  // try again every 2 s and show an error only after 2 failed tries in a row. Events from a socket
  // we closed ourselves (Stop, or a quick Stop / Start) are ignored, so no old error stays on screen.
  useEffect(() => {
    setStats({ fps: 0, latency: 0, process: 0 });   // fresh numbers
    setError(null);
    if (!active) { setStatus("idle"); return undefined; }
    let stopped = false, failures = 0, retry = 0;
    const connect = () => {
      setStatus("connecting");
      const ws = new WebSocket(wsUrl(`/ws/live${REPLAY ? "?replay=1" : ""}`));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      let sentAt = 0, fatal = false;
      ws.onopen = () => { failures = 0; setStatus("open"); setError(null); };
      ws.onclose = () => {
        if (stopped || wsRef.current !== ws) return;          // closed by us
        inFlight.current = false;
        wsRef.current = null;
        if (fatal) { setStatus("error"); return; }            // e.g. the model is missing: retrying will not help
        failures += 1;
        setStatus("connecting");
        if (failures >= 2) setError(NO_BACKEND);
        retry = setTimeout(connect, 2000);
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "frame" || msg.type === "error") {
          inFlight.current = false;
          if (msg.type === "frame") {
            setError(null);                                    // a good frame: an earlier frame error is over
            const now = performance.now();
            const t = timesRef.current;
            t.push(now);
            while (t.length && t[0] < now - 2000) t.shift();
            setStats({ fps: t.length / 2, latency: Math.round(now - sentAt), process: msg.process_ms });
          }
          if (msg.type === "error") {
            setError(msg.message);
            if (msg.fatal) { fatal = true; setStatus("error"); }
          }
        }
        onResultRef.current?.(msg);
      };
      ws._markSent = () => { sentAt = performance.now(); };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      wsRef.current?.close();
      wsRef.current = null;
      inFlight.current = false;
    };
  }, [active]);

  // Capture loop. A timer (not requestAnimationFrame) is used so the rate
  // does not depend on screen refresh; it simply skips while a frame is in flight.
  useEffect(() => {
    if (!active || REPLAY) return undefined;
    let timer = 0;
    let stopped = false;
    const tick = () => {
      if (stopped) return;
      timer = setTimeout(tick, 10);
      const ws = wsRef.current, v = videoRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || inFlight.current || !v || v.readyState < 2) return;
      const now = performance.now();
      if (now - lastSent.current < 1000 / MAX_FPS) return;
      const c = canvasRef.current;
      const scale = SEND_WIDTH / (v.videoWidth || SEND_WIDTH);
      c.width = Math.round((v.videoWidth || 640) * scale);
      c.height = Math.round((v.videoHeight || 480) * scale);
      c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
      inFlight.current = true;
      lastSent.current = now;
      const captured = new Float64Array([now]);       // ms; the server turns it into seconds
      c.toBlob((blob) => {
        if (!blob || ws.readyState !== WebSocket.OPEN) { inFlight.current = false; return; }
        blob.arrayBuffer().then((buf) => { ws._markSent(); ws.send(new Blob([captured, buf])); });
      }, "image/jpeg", 0.8);
    };
    tick();
    return () => { stopped = true; clearTimeout(timer); };
  }, [active, videoRef]);

  return { status, error, stats, send, setError };
}
