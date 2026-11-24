// Left panel: in camera mode a 3D hand that is always on screen: a resting open hand,
// which follows YOUR hand live while the app is reading (the points the server already
// found for the model); in speech-to-fingerspelling mode the reference handshape of the
// letter being played.
import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { HandRig } from "../lib/handRig.js";

// A relaxed open right hand, palm to the camera, as seen in the mirrored video
// (x right, y down, in palm units; the thumb points to the left of the screen).
const REST_HAND = [[0, 0, 0], [-0.1, -0.05, -0.02], [-0.18, -0.12, -0.03], [-0.24, -0.19, -0.04], [-0.29, -0.25, -0.05],
  [-0.08, -0.21, 0], [-0.1, -0.32, -0.01], [-0.11, -0.39, -0.02], [-0.12, -0.45, -0.03],
  [0, -0.22, 0], [0, -0.34, -0.01], [0, -0.42, -0.02], [0, -0.48, -0.03],
  [0.07, -0.2, 0], [0.09, -0.31, -0.01], [0.1, -0.38, -0.02], [0.11, -0.43, -0.03],
  [0.13, -0.17, 0], [0.16, -0.25, -0.01], [0.18, -0.3, -0.02], [0.2, -0.34, -0.03]];

const HandPanel = forwardRef(function HandPanel({ poses, posesError, letter, live, caption, mode }, ref) {
  const canvasRef = useRef(null);
  const rigRef = useRef(null);
  const [error, setError] = useState(null);
  const [shown, setShown] = useState(false);   // a pose has been displayed at least once

  useEffect(() => {
    try {
      rigRef.current = new HandRig(canvasRef.current, { onError: setError });
    } catch {
      setError((e) => e || "3D view unavailable: WebGL could not start in this browser.");
    }
    return () => rigRef.current?.dispose();
  }, []);

  // Expose animation controls to the speech-to-fingerspelling player.
  useImperativeHandle(ref, () => ({
    async show(ch, duration) {
      const p = poses?.[ch];
      if (!p || !rigRef.current) return;
      setShown(true);
      await rigRef.current.setPose(p.landmarks, duration);
    },
    motion: (kind, duration) => rigRef.current?.playMotion(kind, duration) ?? Promise.resolve(),
    stop: () => rigRef.current?.stopMotion(),
  }), [poses]);

  // Camera mode: follow the live hand, or rest (smoothly) when no hand is being followed.
  useEffect(() => {
    if (mode !== "camera" || !rigRef.current) return;
    if (live) rigRef.current.setLive(live);
    else rigRef.current.setLive(REST_HAND, { duration: 350 });
  }, [live, mode]);

  const cameraMode = mode === "camera";
  const ch = cameraMode ? null : letter?.char;
  return (
    <section className="panel hand-panel" aria-label={cameraMode ? "Your hand in 3D" : "Reference handshape"}>
      <header className="panel-head">
        <h2>{cameraMode ? "Your hand (live 3D)" : "Reference handshape"}</h2>
        <button className="btn ghost small" onClick={() => rigRef.current?.resetView()}
          title="Reset the 3D view">Reset view</button>
      </header>
      <div className="hand-stage">
        <canvas ref={canvasRef} aria-label={ch ? `3D hand showing ASL letter ${ch}` : "3D hand"} />
        {!cameraMode && !ch && !shown && !error && <div className="stage-hint">{caption}</div>}
        {(error || (!cameraMode && posesError)) && <div className="stage-error">{error || posesError}</div>}
      </div>
      {cameraMode ? (
        <footer className="hand-foot">
          <span className="meta-line muted">{live ? "Following your hand" : caption} · drag to rotate</span>
        </footer>
      ) : (
      <footer className="hand-foot">
        <div className="hand-letter" aria-live="polite">{ch && ch.length === 1 ? ch : "–"}</div>
      </footer>
      )}
    </section>
  );
});

export default HandPanel;
