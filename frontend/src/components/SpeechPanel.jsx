// Right panel (speech mode): "Speech to ASL fingerspelling".
// Microphone -> local Vosk transcript -> user corrects it -> the 3D hand
// fingerspells it letter by letter. This is NOT full ASL translation:
// it only spells English letters A–Z.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { planFingerspelling, durations } from "../lib/fingerspell.js";
import { useSpeechInput } from "../lib/useSpeechInput.js";

const sleep = (ms, token) => new Promise((res) => {
  const t = setTimeout(res, ms);
  token.cancel = () => { clearTimeout(t); res(); };
});

export default function SpeechPanel({ handRef, poses, speechStatus, onPlaying }) {
  const [transcript, setTranscript] = useState("");
  const [plan, setPlan] = useState(null);
  const [index, setIndex] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const runRef = useRef({ id: 0, paused: false, token: {} });
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const [elapsed, setElapsed] = useState(0);

  const mic = useSpeechInput({ onFinal: (t) => setTranscript((prev) => (prev ? `${prev} ${t}` : t).trim()) });

  useEffect(() => {
    if (!mic.recording) return undefined;
    const iv = setInterval(() => setElapsed(Math.floor((Date.now() - mic.startedAt) / 1000)), 250);
    return () => clearInterval(iv);
  }, [mic.recording, mic.startedAt]);

  const letters = useMemo(() => Object.keys(poses || {}), [poses]);

  const stopPlayback = useCallback(() => {
    runRef.current.id++;
    runRef.current.token.cancel?.();
    handRef.current?.stop();
    setPlaying(false);
  }, [handRef]);

  useEffect(() => () => stopPlayback(), [stopPlayback]);

  const play = useCallback(async (p, from = 0) => {
    const run = ++runRef.current.id;
    runRef.current.paused = false;
    setPlaying(true);
    const alive = () => runRef.current.id === run;
    for (let i = from; i < p.items.length && alive(); i++) {
      const it = p.items[i];
      setIndex(i);
      onPlaying?.(it.type === "letter" ? it.char : null);
      const d = durations(speedRef.current);
      const token = (runRef.current.token = {});
      if (it.type === "letter") {
        if (it.repeat) await handRef.current?.motion("slide", d.repeatSlide);   // repeated-letter cue
        if (!alive()) return;
        await handRef.current?.show(it.char, d.transition);
        if (!alive()) return;
        if (it.char === "J" || it.char === "Z") await handRef.current?.motion(it.char, d.motion);
        await sleep(d.hold, token);
      } else if (it.type === "pause") {
        await sleep(d.wordPause, token);
      } else {
        await sleep(d.hold * 0.8, token);   // show the unsupported mark briefly, do not sign
      }
    }
    if (alive()) { setPlaying(false); setIndex(-1); onPlaying?.(null); }
  }, [handRef, onPlaying]);

  const fingerspell = () => {
    stopPlayback();
    const p = planFingerspelling(transcript, letters);
    setPlan(p);
    setIndex(-1);
    if (p.letters) play(p, 0);
  };

  const pause = () => { runRef.current.pausedAt = index; stopPlayback(); };
  const resume = () => plan && play(plan, Math.max(0, runRef.current.pausedAt ?? 0));
  const replay = () => plan && (stopPlayback(), play(plan, 0));
  const clear = () => { stopPlayback(); setTranscript(""); setPlan(null); setIndex(-1); onPlaying?.(null); };

  const engineOk = speechStatus?.available;
  return (
    <section className="panel speech-panel" aria-label="Speech to ASL fingerspelling">
      <header className="panel-head">
        <h2>Speech to ASL fingerspelling</h2>
        <span className={`badge ${engineOk ? "verified" : "unverified"}`}
          title={speechStatus?.message}>{engineOk ? "Local engine ready" : "Speech engine not installed"}</span>
      </header>
      <div className="speech-body">
        <p className="muted small-print">
          Spells English letters A–Z with the reference handshapes. It is not a full ASL
          translation. Recognition runs locally (Vosk); audio is not sent anywhere else.
        </p>
        <div className="mic-row">
          {!mic.recording ? (
            <button className="btn primary" onClick={mic.start} disabled={!engineOk}
              title={engineOk ? "Start recording" : speechStatus?.message}>● Start microphone</button>
          ) : (
            <button className="btn danger" onClick={mic.stop}>■ Stop microphone</button>
          )}
          {mic.recording && (
            <span className="rec-indicator" role="status">
              <i className="rec-dot" /> Recording {elapsed}s
              <span className="level"><i style={{ width: `${Math.min(100, mic.level * 160)}%` }} /></span>
            </span>
          )}
        </div>
        {!engineOk && speechStatus?.message && <p className="err-text">{speechStatus.message}. You can still type text below.</p>}
        {mic.error && <p className="err-text">{mic.error}</p>}
        {mic.partial && <p className="partial">Hearing: “{mic.partial}”</p>}
        <label className="field">
          <span>Transcript (check and correct before fingerspelling)</span>
          <textarea rows={3} value={transcript} onChange={(e) => setTranscript(e.target.value)}
            placeholder="Speak, or type words here…" />
        </label>
        <div className="player-row">
          <button className="btn primary" onClick={fingerspell} disabled={!transcript.trim() || !letters.length}>
            Fingerspell
          </button>
          {playing ? (
            <button className="btn" onClick={pause}>Pause</button>
          ) : (
            <button className="btn" onClick={resume} disabled={!plan?.letters}>Play</button>
          )}
          <button className="btn" onClick={replay} disabled={!plan?.letters}>Replay</button>
          <button className="btn" onClick={clear} disabled={!transcript && !plan}>Clear</button>
          <label className="rate">
            <span>Speed {speed}×</span>
            <input type="range" min="0.5" max="2" step="0.25" value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))} />
          </label>
        </div>
        {plan && (
          <>
            <div className="sequence" aria-label="Fingerspelling sequence">
              {plan.items.map((it, i) => (
                it.type === "pause" ? <span key={i} className="seq-gap" aria-label="word break" /> :
                  <span key={i} className={`seq-chip ${it.type} ${i === index ? "now" : ""} ${it.repeat ? "rep" : ""}`}
                    title={it.type === "unsupported" ? "Not supported: not signed" : it.repeat ? "Repeated letter" : ""}>
                    {it.char}
                  </span>
              ))}
            </div>
            {plan.unsupported.length > 0 && (
              <p className="warn-text">
                Not signed (unsupported in this release): {plan.unsupported.map((c) => `“${c}”`).join(" ")}
              </p>
            )}
          </>
        )}
      </div>
    </section>
  );
}
