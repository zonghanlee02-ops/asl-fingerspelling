// Right panel (camera mode): live video, optional landmark overlay, and the words
// final_model_i has read so far (confirmed words, then the current guess, shown alike).
// Captions are NOT drawn on the video; they live in the separate caption bar.
import React, { useEffect, useRef } from "react";

const CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

export const STATE_INFO = {
  idle: { label: "Camera off", tone: "neutral" },
  ready: { label: "Press Start to read your signing", tone: "neutral" },
  no_hand: { label: "No hand detected", tone: "neutral" },
  reading: { label: "Reading your signing…", tone: "info" },
};

export default function CameraPanel({ cam, reading, result, overlay, setOverlay, displayState, stats,
  connection }) {
  const boxRef = useRef(null);
  const canvasRef = useRef(null);

  // Draw landmarks on a canvas sized to the visible video area.
  useEffect(() => {
    const c = canvasRef.current, box = boxRef.current, v = cam.videoRef.current;
    if (!c || !box || !v) return;
    const rect = box.getBoundingClientRect();
    const ar = cam.aspect;
    let w = rect.width, h = rect.width / ar;
    if (h > rect.height) { h = rect.height; w = h * ar; }
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    c.width = Math.round(w * devicePixelRatio);
    c.height = Math.round(h * devicePixelRatio);
    const g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height);
    const lm = result?.hand?.map(([x, y]) => [x, y * ar]);    // the server sends y in width units
    if (!overlay || !lm || !reading) return;
    g.scale(devicePixelRatio, devicePixelRatio);
    g.lineWidth = 2.5;
    g.strokeStyle = "rgba(255,255,255,0.85)";
    for (const [a, b] of CONNECTIONS) {
      g.beginPath();
      g.moveTo(lm[a][0] * w, lm[a][1] * h);
      g.lineTo(lm[b][0] * w, lm[b][1] * h);
      g.stroke();
    }
    g.fillStyle = "#0a84ff";
    for (const [x, y] of lm) { g.beginPath(); g.arc(x * w, y * h, 3.5, 0, 7); g.fill(); }
  }, [result, overlay, reading, cam.aspect, cam.videoRef]);

  const info = STATE_INFO[displayState] || STATE_INFO.idle;
  return (
    <section className="panel camera-panel" aria-label="Live camera">
      <header className="panel-head">
        <h2>Live camera</h2>
        <div className="head-tools">
          <label className="toggle small">
            <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
            <span>Landmarks</span>
          </label>
        </div>
      </header>
      <div className="video-box" ref={boxRef}>
        <video ref={cam.videoRef} className="mirror" playsInline muted
          style={{ aspectRatio: cam.aspect }} aria-label="Your camera (mirrored)" />
        <canvas ref={canvasRef} className="overlay" />
        {!cam.running && !reading && (
          <div className="stage-hint">
            {cam.error ? <span className="err-text">{cam.error}</span> : "Turning the camera on…"}
          </div>
        )}
        <span className={`state-pill ${info.tone}`} role="status">{info.label}</span>
      </div>
      <footer className="prediction">
        <div className="pred-body" aria-live="polite" aria-label="Words being read">
          <div className="live-words">
            {[result?.confirmed, result?.current].filter(Boolean).join(" ")}
            {!result?.confirmed && !result?.current && <span className="muted">Words appear here while you sign.</span>}
          </div>
        </div>
        <div className="pred-stats" title="Frames read per second, round trip, and server time per frame">
          <span>{connection}</span>
          <span>{stats.fps.toFixed(0)} fps</span>
          <span>{stats.latency} ms</span>
          <span>server {result?.process_ms ?? "–"} ms</span>
          <span className="muted">{result?.model || ""}</span>
        </div>
      </footer>
    </section>
  );
}
