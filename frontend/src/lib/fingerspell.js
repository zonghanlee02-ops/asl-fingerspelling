// Turn text into an ordered fingerspelling plan for the 3D hand.
//
// Rules (first release):
//  * Supported: letters A–Z that have a reference pose.
//  * Spaces / line breaks mark word boundaries (a longer pause).
//  * A letter equal to the previous letter in the same word is a repeated
//    letter (e.g. the LL in HELLO): the hand shows a small slide between them.
//  * Anything else (digits, punctuation, accented letters, emoji) is kept in
//    the plan as "unsupported": it is shown to the user and NOT signed.
//    Nothing is invented or silently dropped.

export function planFingerspelling(text, availableLetters) {
  const have = new Set(availableLetters);
  const items = [];
  const unsupported = [];
  const words = String(text).trim().split(/\s+/).filter(Boolean);
  words.forEach((word, w) => {
    if (w > 0) items.push({ type: "pause" });
    let prev = null;
    for (const ch of Array.from(word)) {          // Array.from keeps emoji whole
      const up = ch.toUpperCase();
      if (/^[A-Z]$/.test(up) && have.has(up)) {
        items.push({ type: "letter", char: up, word: w, repeat: prev === up });
        prev = up;
      } else {
        items.push({ type: "unsupported", char: ch, word: w });
        if (!unsupported.includes(ch)) unsupported.push(ch);
        prev = null;
      }
    }
  });
  const letters = items.filter((i) => i.type === "letter").length;
  return { items, unsupported, letters, words: words.length };
}

// Durations in milliseconds at speed 1.0.
export const TIMING = { transition: 260, hold: 650, repeatSlide: 260, wordPause: 900, motion: 900 };

export function durations(speed) {
  const s = Math.max(0.25, Number(speed) || 1);
  return Object.fromEntries(Object.entries(TIMING).map(([k, v]) => [k, v / s]));
}
