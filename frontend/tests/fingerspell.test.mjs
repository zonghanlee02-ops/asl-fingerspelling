// Run with:  npm test   (inside frontend/)
import { test } from "node:test";
import assert from "node:assert/strict";
import { planFingerspelling } from "../src/lib/fingerspell.js";

const AZ = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
const seq = (p) => p.items.map((i) => (i.type === "letter" ? i.char + (i.repeat ? "*" : "") :
  i.type === "pause" ? "_" : `!${i.char}`)).join("");

test("letters in order with word pauses", () => {
  assert.equal(seq(planFingerspelling("hello world", AZ)), "HEL" + "L*" + "O_WORLD");
});

test("repeated letters are kept and marked", () => {
  const p = planFingerspelling("BOOK", AZ);
  assert.equal(seq(p), "BOO*K");
  assert.equal(p.letters, 4);
});

test("unsupported characters are listed, not invented", () => {
  const p = planFingerspelling("Room 3B, café!", AZ);
  assert.equal(seq(p), "ROO*M_!3B!,_CAF!é!!");
  assert.deepEqual(p.unsupported, ["3", ",", "é", "!"]);
});

test("a letter without a pose is unsupported", () => {
  const p = planFingerspelling("AZ", AZ.filter((c) => c !== "Z"));
  assert.deepEqual(p.unsupported, ["Z"]);
});

test("extra whitespace does not create extra pauses", () => {
  assert.equal(seq(planFingerspelling("  hi \n  there ", AZ)), "HI_THERE");
});
