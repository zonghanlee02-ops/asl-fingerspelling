// Camera access with clear error messages.
import { useCallback, useEffect, useRef, useState } from "react";

export function cameraErrorMessage(err) {
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Allow camera access for this page in the browser's site settings, then press Turn camera on.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found. Connect a webcam and press Turn camera on.";
    case "NotReadableError":
    case "AbortError":
      return "The camera is busy or could not start. Close other apps using the camera (Zoom, Teams, the old app.py window) and try again.";
    default:
      if (!navigator.mediaDevices?.getUserMedia)
        return "This browser cannot access cameras here. Open the app at http://localhost or http://127.0.0.1 in Chrome, Edge or Firefox.";
      return `Camera error: ${err?.message || err}`;
  }
}

export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [aspect, setAspect] = useState(16 / 9);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: "user" },  // as your recordings
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      v.srcObject = stream;
      await v.play();
      if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
      stream.getVideoTracks()[0].addEventListener("ended", () => {
        setError("The camera stopped (unplugged or taken by another app).");
        stop();
      });
      setRunning(true);
    } catch (e) {
      setError(cameraErrorMessage(e));
      stop();
    }
  }, [stop]);

  useEffect(() => stop, [stop]);
  return { videoRef, running, error, aspect, start, stop, setError };
}
