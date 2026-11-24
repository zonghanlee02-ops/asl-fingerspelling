// App.jsx - full-window layout and the glue between camera, backend,
// captions, speech output and the 3D hand.
//
// Live screen, camera recognition: the camera turns on by itself; Start begins reading with
// final_model_i (continuous fingerspelling: TCN + CTC + word decoder, over /ws/live) and Stop
// pauses it.
// The 3D hand is always on screen: a resting open hand that follows your hand while reading.
// Keys on the camera screen: S start / stop, Delete, C clear, F finish (not while typing in a box).
// The A symbol opens the side strip (like Chrome's vertical tabs): Live and ASL Alphabet.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./lib/api.js";
import { useCamera } from "./lib/useCamera.js";
import { REPLAY, useRecognition } from "./lib/useRecognition.js";
import { useSpeechSynthesis } from "./lib/useSpeechSynthesis.js";
import HandPanel from "./components/HandPanel.jsx";
import CameraPanel from "./components/CameraPanel.jsx";
import SpeechPanel from "./components/SpeechPanel.jsx";
import CaptionBar from "./components/CaptionBar.jsx";
import VoiceSettings from "./components/VoiceSettings.jsx";
import AlphabetScreen from "./components/AlphabetScreen.jsx";

const SCREENS = [{ id: "live", icon: "📷", label: "Live" }, { id: "alphabet", icon: "🤟", label: "ASL Alphabet" }];
const Key = ({ k }) => <kbd className="key">{k}</kbd>;

const cleanText = (v) => String(v).toUpperCase().split(/\s+/).filter(Boolean).join(" ");

export default function App() {
  const [view, setView] = useState("live");            // live | alphabet (side strip)
  const [sideOpen, setSideOpen] = useState(false);
  const [liveMode, setLiveMode] = useState("camera");  // camera | speech
  const [reading, setReading] = useState(REPLAY);      // Start / Stop
  const [health, setHealth] = useState(null);
  const [backendErr, setBackendErr] = useState(null);
  const [poses, setPoses] = useState(null);
  const [posesErr, setPosesErr] = useState(null);
  const [result, setResult] = useState(null);
  const [sentences, setSentences] = useState([]);      // finished sentences
  const [overlay, setOverlay] = useState(true);
  const [playingLetter, setPlayingLetter] = useState(null);
  const handRef = useRef(null);
  const spoken = useRef(new Set());

  const cam = useCamera();
  const tts = useSpeechSynthesis();
  const ttsRef = useRef(tts);
  ttsRef.current = tts;

  useEffect(() => {
    let timer = 0;    // backend still starting: ask again every 3 s, and the message goes away when it answers
    const check = () => api("/api/health").then((h) => { setHealth(h); setBackendErr(null); })
      .catch((e) => { setBackendErr(e.message); timer = setTimeout(check, 3000); });
    check();
    fetch("/reference_poses.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((d) => setPoses(d.poses))
      .catch(() => setPosesErr("Reference poses could not be loaded (frontend/public/reference_poses.json). Run scripts/app/build_reference_poses.py."));
    return () => clearTimeout(timer);
  }, []);

  // A finished sentence: store it and speak it once.
  const finishSentence = useCallback((id, t) => {
    t = (t || "").trim();
    if (!t) return;
    setSentences((s) => [...s, t]);
    ttsRef.current.setLastText(t);
    if (ttsRef.current.settings.enabled && !spoken.current.has(id)) {
      spoken.current.add(id);
      ttsRef.current.speak(t);
    }
  }, []);

  const onResult = useCallback((msg) => {
    if (msg.type === "frame") setResult(msg);
    if (msg.type === "text" && msg.finished_text !== undefined) {         // after Finish / Clear
      setResult((r) => r && { ...r, confirmed: "", current: "" });
      finishSentence(`seq-${Date.now()}`, msg.finished_text);
    }
  }, [finishSentence]);

  // The camera turns on by itself (once) on entering camera mode, and off anywhere else.
  const cameraScreen = view === "live" && liveMode === "camera";
  const autoStarted = useRef(false);
  useEffect(() => {
    if (cameraScreen && !autoStarted.current && !REPLAY) { autoStarted.current = true; cam.start(); }
    if (!cameraScreen) {
      autoStarted.current = false;
      setReading(false);
      if (cam.running) cam.stop();
    }
  }, [cameraScreen, cam]);

  const active = cameraScreen && reading && (cam.running || REPLAY);
  const rec = useRecognition({ videoRef: cam.videoRef, active, onResult });
  useEffect(() => { if (!active) setResult(null); }, [active]);

  // Delete and Edit act on the last finished sentence (words being read can still change).
  const editLast = (fn) => setSentences((s) => {
    if (!s.length) return s;
    const t = cleanText(fn(s[s.length - 1]));
    if (t) ttsRef.current.setLastText(t);
    return t ? [...s.slice(0, -1), t] : s.slice(0, -1);
  });

  const command = (action, value) => {
    if (action === "delete") return editLast((t) => t.split(" ").slice(0, -1).join(" "));
    if (action === "set_text") return editLast(() => value);
    rec.send({ type: "command", action });                         // finish | clear (while reading)
  };

  // Keyboard: S start / stop, Delete, C clear, F finish. Ignored while typing in a text box.
  const keysRef = useRef(null);
  keysRef.current = {
    s: () => (reading ? setReading(false) : (cam.running || REPLAY) && setReading(true)),
    delete: () => command("delete"),
    c: () => command("clear"),
    f: () => active && command("finish"),
  };
  useEffect(() => {
    if (!cameraScreen) return undefined;
    const onKey = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || e.target.closest?.("input, textarea, select, [contenteditable]")) return;
      const run = keysRef.current[e.key.toLowerCase()];
      if (run) { e.preventDefault(); run(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cameraScreen]);

  const live = active ? result?.hand : null;
  const displayState = !cam.running && !REPLAY ? "idle" : !reading ? "ready" : live ? "reading" : "no_hand";
  const liveText = [result?.confirmed, result?.current].filter(Boolean).join(" ");

  const errors = [...new Set([backendErr, rec.error, tts.error,
    health && !health.live_model?.found && `The model ${health.live_model?.name} is missing in backend/models.`].filter(Boolean))];

  return (
    <div className="shell">
    <aside className={`sidebar ${sideOpen ? "open" : ""}`} aria-label="Screens">
      <button className="logo" onClick={() => setSideOpen((o) => !o)} aria-expanded={sideOpen}
        title={sideOpen ? "Close the side bar" : "Open the side bar"}>A</button>
      {SCREENS.map((s) => (
        <button key={s.id} className={`side-item ${view === s.id ? "on" : ""}`} onClick={() => setView(s.id)} title={s.label}>
          <span className="side-icon" aria-hidden="true">{s.icon}</span><span className="side-label">{s.label}</span>
        </button>
      ))}
    </aside>
    <div className={`app view-${view}`}>
      <header className="topbar">
        <div className="brand">
          <div><h1>{view === "alphabet" ? "ASL Alphabet" : "ASL Fingerspelling"}</h1>
            <span className="sub">{view === "alphabet" ? "Every reference handshape in 3D · ← → or a letter key" : "Local · your own trained model"}</span></div>
        </div>
        {view === "live" && <div className="segmented" role="tablist" aria-label="Input mode">
          <button role="tab" aria-selected={liveMode === "camera"} className={liveMode === "camera" ? "on" : ""}
            onClick={() => setLiveMode("camera")}>Camera recognition</button>
          <button role="tab" aria-selected={liveMode === "speech"} className={liveMode === "speech" ? "on" : ""}
            onClick={() => setLiveMode("speech")}>Speech to ASL fingerspelling</button>
        </div>}
        <nav className="nav" aria-label="Screens">
          <button className="icon" title="Full screen" aria-label="Toggle full screen"
            onClick={() => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.())}>⤢</button>
        </nav>
      </header>

      {errors.length > 0 && (
        <div className="banner error" role="alert">
          {errors.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}

      {view === "alphabet" ? <AlphabetScreen /> : <>
      <main className="stage">
        <HandPanel ref={handRef} poses={poses} posesError={posesErr} mode={liveMode} live={live}
          letter={liveMode === "speech" && playingLetter ? { char: playingLetter } : null}
          caption={liveMode === "camera" ? (active ? "Show your hand to the camera" : "Press Start, then sign")
            : "Press Fingerspell to play the transcript."} />
        {liveMode === "camera" ? (
          <CameraPanel cam={cam} reading={active} result={active ? result : null} overlay={overlay} setOverlay={setOverlay}
            displayState={displayState} stats={rec.stats} connection={rec.status} />
        ) : (
          <SpeechPanel handRef={handRef} poses={poses} speechStatus={health?.speech} onPlaying={setPlayingLetter} />
        )}
      </main>
      {liveMode === "camera" && (
        <CaptionBar text={liveText} sentences={sentences}
          onEdit={sentences.length ? (v) => command("set_text", v) : undefined}
          editValue={sentences[sentences.length - 1]}
          placeholder={reading ? "Fingerspell…" : "Press Start to fingerspell."} />
      )}
      <footer className="controls" aria-label="Controls">
        {liveMode === "camera" && (
          <div className="ctrl-group">
            {!cam.running && !REPLAY && <button className="btn" onClick={cam.start}>Turn camera on</button>}
            {!reading
              ? <button className="btn primary" onClick={() => setReading(true)} disabled={!cam.running && !REPLAY}>Start <Key k="S" /></button>
              : <button className="btn" onClick={() => setReading(false)}>Stop <Key k="S" /></button>}
            <button className="btn" onClick={() => command("delete")}>Delete <Key k="Del" /></button>
            <button className="btn" onClick={() => command("clear")}>Clear <Key k="C" /></button>
            <button className="btn accent" onClick={() => command("finish")}
              disabled={!active}>Finish sentence <Key k="F" /></button>
          </div>
        )}
        <VoiceSettings tts={tts} />
        {!tts.supported && <span className="muted">Speech output not supported in this browser.</span>}
      </footer>
      </>}
    </div>
    </div>
  );
}
