// ASL Alphabet screen (side bar): every reference handshape A-Z in 3D. Step through
// the letters and compare each pose with a reliable ASL fingerspelling chart before
// marking it verified with scripts/app/verify_poses.py. Nothing here changes the poses.
import React, { useEffect, useRef, useState } from "react";
import { HandRig } from "../lib/handRig.js";

export default function AlphabetScreen() {
  const canvasRef = useRef(null);
  const rigRef = useRef(null);
  const [poses, setPoses] = useState(null);
  const [letter, setLetter] = useState("A");
  const [err, setErr] = useState(null);

  useEffect(() => {
    try { rigRef.current = new HandRig(canvasRef.current); } catch { setErr("WebGL is not available."); }
    fetch("/reference_poses.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => setPoses(d.poses))
      .catch(() => setErr("Could not load reference_poses.json"));
    return () => rigRef.current?.dispose();
  }, []);

  useEffect(() => {
    const p = poses?.[letter];
    if (!p || !rigRef.current) return;
    rigRef.current.setPose(p.landmarks, 300).then(() => {
      if (p.motion) rigRef.current.playMotion(letter, 1000);
    });
  }, [letter, poses]);

  useEffect(() => {
    const keys = Object.keys(poses || {});
    const onKey = (e) => {
      const i = keys.indexOf(letter);
      if (e.key === "ArrowRight") setLetter(keys[(i + 1) % keys.length]);
      if (e.key === "ArrowLeft") setLetter(keys[(i - 1 + keys.length) % keys.length]);
      if (/^[a-z]$/i.test(e.key) && keys.includes(e.key.toUpperCase())) setLetter(e.key.toUpperCase());
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [letter, poses]);

  const p = poses?.[letter];
  const v = p?.verification || {};
  return (
    <>
      <div className="letters" role="group" aria-label="Letters">
        {Object.entries(poses || {}).map(([L, q]) => (
          <button key={L} className={`letter-btn ${L === letter ? "on" : ""} ${q.verification?.status} ${q.dev_check?.result === "flagged" && q.verification?.status !== "verified" ? "flagged" : ""}`} onClick={() => setLetter(L)}>{L}</button>
        ))}
      </div>
      <main className="stage review-stage">
        <section className="panel hand-panel">
          <header className="panel-head"><h2>Letter {letter}</h2>
            <button className="btn ghost small" onClick={() => rigRef.current?.resetView()}>Reset view</button></header>
          <div className="hand-stage"><canvas ref={canvasRef} />{err && <div className="stage-error">{err}</div>}</div>
        </section>
        <section className="panel pad review-info">
          <div className="hand-letter">{letter}</div>
          <p><span className={`badge ${v.status || "unverified"}`}>{v.status || "unverified"}</span>
            {v.reference && <span className="muted"> · {v.reference} · {v.by} · {v.date}</span>}</p>
          <p className="small-print">{p?.source}</p>
          {p?.dev_check?.result === "flagged" && v.status !== "verified" && <p className="err-text">Flagged during development: {p.dev_check.notes}</p>}
          {p?.motion && <p className="warn-text">Motion letter: {p.motion} (path added from the written description, not from data).</p>}
          {v.notes && <p>Notes: {v.notes}</p>}
          <h3>How to verify</h3>
          <ol className="small-print">
            <li>Open a reliable ASL fingerspelling reference (for example a university or Deaf-led ASL course chart).</li>
            <li>Rotate the 3D hand (drag) and compare the handshape: which fingers are extended, thumb position, palm direction.</li>
            <li>Record the result:<br />
              <code>python scripts/app/verify_poses.py --letters {letter} --by "Your name" --reference "Chart name"</code><br />
              or add <code>--reject --notes "what is wrong"</code>.</li>
          </ol>
          <p className="muted small-print">Use ← → or press a letter key to move between poses.</p>
        </section>
      </main>
    </>
  );
}
