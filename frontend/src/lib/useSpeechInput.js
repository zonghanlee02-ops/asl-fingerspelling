// Microphone -> local Vosk engine (through the Python backend).
// Recording starts ONLY when the user presses the microphone button.
// No cloud service is used; if the local engine is missing we say so.
import { useCallback, useRef, useState } from "react";
import { api, wsUrl } from "./api.js";

function micError(e) {
  if (e?.name === "NotAllowedError" || e?.name === "SecurityError")
    return "Microphone permission was denied. Allow microphone access for this page in the browser's site settings.";
  if (e?.name === "NotFoundError") return "No microphone was found.";
  if (e?.name === "NotReadableError") return "The microphone is busy or could not start.";
  return `Microphone error: ${e?.message || e}`;
}

export function useSpeechInput({ onFinal }) {
  const [recording, setRecording] = useState(false);
  const [partial, setPartial] = useState("");
  const [error, setError] = useState(null);
  const [level, setLevel] = useState(0);
  const [startedAt, setStartedAt] = useState(null);
  const res = useRef({});

  const cleanup = useCallback(() => {
    const r = res.current;
    r.stream?.getTracks().forEach((t) => t.stop());
    r.node?.disconnect();
    r.ctx?.close().catch(() => {});
    res.current = {};
    setRecording(false);
    setLevel(0);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setPartial("");
    try {
      const st = await api("/api/speech/status");
      if (!st.available) { setError(`Local speech recognition unavailable: ${st.message}`); return; }
    } catch (e) { setError(e.message); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const ctx = new AudioContext();
      await ctx.audioWorklet.addModule("/pcm-worklet.js");
      const src = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "pcm-worklet");
      const ws = new WebSocket(wsUrl("/ws/speech"));
      ws.binaryType = "arraybuffer";
      // Audio captured before the engine is ready is buffered, not lost.
      res.current = { stream, ctx, node, ws, ready: false, pending: [] };
      node.port.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          if (res.current.ready && ws.readyState === WebSocket.OPEN) ws.send(ev.data);
          else if (res.current.pending) res.current.pending.push(ev.data);
        } else if (ev.data?.level !== undefined) {
          setLevel(ev.data.level);
        }
      };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        if (m.type === "ready") {
          res.current.pending.forEach((buf) => ws.send(buf));
          res.current.pending = null;
          res.current.ready = true;
        }
        else if (m.type === "partial") setPartial([m.transcript, m.text].filter(Boolean).join(" "));
        else if (m.type === "final_part") setPartial(m.transcript);
        else if (m.type === "final") { onFinal?.(m.transcript); setPartial(""); cleanup(); }
        else if (m.type === "error") { setError(m.message); cleanup(); }
      };
      ws.onerror = () => { setError("Speech connection to the local backend failed."); cleanup(); };
      ws.onclose = () => { if (res.current.ws === ws) cleanup(); };
      src.connect(node);   // the worklet does not output sound, nothing is played back
      setRecording(true);  // the microphone is live from here: show the indicator now
      setStartedAt(Date.now());
    } catch (e) {
      setError(micError(e));
      cleanup();
    }
  }, [cleanup, onFinal]);

  const stop = useCallback(() => {
    const { ws, stream } = res.current;
    stream?.getTracks().forEach((t) => t.stop());   // stop the mic immediately
    setRecording(false);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
    else cleanup();
  }, [cleanup]);

  return { recording, partial, error, setError, level, startedAt, start, stop };
}
