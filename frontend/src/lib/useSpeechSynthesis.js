// Text-to-speech with the browser's built-in speechSynthesis.
// Only English voices are offered. Nothing is spoken automatically except
// a finished sentence (handled in App.jsx), and only when enabled.
import { useCallback, useEffect, useRef, useState } from "react";

const KEY = "asl.tts";

function loadSettings() {
  try {
    return { enabled: false, voiceURI: "", rate: 1, ...JSON.parse(localStorage.getItem(KEY) || "{}") };
  } catch {
    return { enabled: false, voiceURI: "", rate: 1 };
  }
}

export function useSpeechSynthesis() {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [voices, setVoices] = useState([]);
  const [settings, setSettings] = useState(loadSettings);
  const [speaking, setSpeaking] = useState(false);
  const [lastText, setLastText] = useState("");
  const [error, setError] = useState(null);
  const utterRef = useRef(null);

  useEffect(() => {
    if (!supported) return undefined;
    const load = () => setVoices(speechSynthesis.getVoices().filter((v) => /^en(-|_|$)/i.test(v.lang)));
    load();
    speechSynthesis.addEventListener("voiceschanged", load);
    return () => speechSynthesis.removeEventListener("voiceschanged", load);
  }, [supported]);

  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(settings)); } catch { /* private mode */ }
  }, [settings]);

  const speak = useCallback((text) => {
    const t = String(text || "").trim();
    if (!t) return false;
    setLastText(t);
    if (!supported) { setError("This browser has no speech synthesis."); return false; }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(t);
    const v = voices.find((x) => x.voiceURI === settings.voiceURI) || voices[0];
    if (v) u.voice = v;
    u.lang = v?.lang || "en-US";
    u.rate = settings.rate;
    u.onstart = () => setSpeaking(true);
    u.onend = () => setSpeaking(false);
    u.onerror = (e) => {
      setSpeaking(false);
      if (e.error !== "canceled" && e.error !== "interrupted") setError(`Speech failed: ${e.error}`);
    };
    utterRef.current = u;
    speechSynthesis.speak(u);
    return true;
  }, [supported, voices, settings]);

  const stop = useCallback(() => {
    if (supported) speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  const update = useCallback((patch) => setSettings((s) => ({ ...s, ...patch })), []);

  return { supported, voices, settings, update, speak, stop, speaking, lastText, setLastText, error, setError };
}
