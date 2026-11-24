// Text-to-speech controls: enable toggle, voice, rate, speak again, stop.
import React from "react";

export default function VoiceSettings({ tts }) {
  const { settings, update, voices, supported } = tts;
  return (
    <div className="voice-group" role="group" aria-label="Speech output">
      <label className="toggle">
        <input type="checkbox" checked={settings.enabled} disabled={!supported}
          onChange={(e) => { update({ enabled: e.target.checked }); if (!e.target.checked) tts.stop(); }} />
        <span>Enable speech</span>
      </label>
      <select aria-label="Voice" value={settings.voiceURI} disabled={!supported || !voices.length}
        onChange={(e) => update({ voiceURI: e.target.value })}>
        {!voices.length && <option value="">No English voice found</option>}
        {voices.map((v) => <option key={v.voiceURI} value={v.voiceURI}>{v.name} ({v.lang})</option>)}
      </select>
      <label className="rate" title="Speaking rate">
        <span>Rate {settings.rate.toFixed(1)}×</span>
        <input type="range" min="0.5" max="1.6" step="0.1" value={settings.rate}
          onChange={(e) => update({ rate: Number(e.target.value) })} />
      </label>
      <button className="btn small" disabled={!tts.lastText} onClick={() => tts.speak(tts.lastText)}
        title={tts.lastText ? `Speak: "${tts.lastText}"` : "Finish a sentence first"}>Speak again</button>
      <button className="btn small" disabled={!tts.speaking} onClick={tts.stop}>Stop speaking</button>
    </div>
  );
}
