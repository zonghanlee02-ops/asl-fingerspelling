// Bottom-centre caption area: finished sentences + the sentence being built.
// The user can correct the text with Edit; we never auto-correct it.
import React, { useEffect, useRef, useState } from "react";

// editValue: what Edit opens (default: the sentence being built). With final_model_i the
// words being read can still change, so Edit corrects the last finished sentence instead.
export default function CaptionBar({ text, sentences, finished, onEdit, placeholder, editValue }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);
  useEffect(() => { scrollRef.current?.scrollTo({ top: 1e6 }); }, [text, sentences]);

  const save = () => { onEdit(draft); setEditing(false); };
  return (
    <section className="caption" aria-label="Recognised text">
      <div className="caption-text" ref={scrollRef} aria-live="polite">
        {sentences.slice(-2).map((s, i) => <p key={i} className="done-line">{s}</p>)}
        {editing ? (
          <form onSubmit={(e) => { e.preventDefault(); save(); }} className="edit-row">
            <input ref={inputRef} value={draft} onChange={(e) => setDraft(e.target.value)}
              aria-label="Edit recognised text" onKeyDown={(e) => e.key === "Escape" && setEditing(false)} />
            <button className="btn primary small" type="submit">Save</button>
            <button className="btn ghost small" type="button" onClick={() => setEditing(false)}>Cancel</button>
          </form>
        ) : (
          <p className="live-line">
            {text ? text : <span className="muted">{finished ? "Sentence finished." : placeholder}</span>}
            <span className="cursor" aria-hidden="true" />
          </p>
        )}
      </div>
      {!editing && onEdit && (
        <button className="btn ghost small edit-btn" onClick={() => { setDraft(editValue ?? text); setEditing(true); }}>
          Edit
        </button>
      )}
    </section>
  );
}
