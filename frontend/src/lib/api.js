// Small helpers for talking to the local Python backend.

export const NO_BACKEND = "Cannot reach the local backend. Start it (in PowerShell: .\\start.bat); the app reconnects by itself.";

export async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method, headers: {} };
  if (form) {
    opts.body = form;                     // multipart (file upload)
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error(NO_BACKEND);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : `Request failed (${res.status})`);
  }
  return data;
}

// ws://host/... for the same host the page was loaded from (works with the
// Vite dev proxy and with the Python server serving the built app).
export function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

export const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);
