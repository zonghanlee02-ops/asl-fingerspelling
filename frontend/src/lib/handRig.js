// handRig.js - An ORIGINAL procedural 3D hand built in code with Three.js.
//
// No downloaded model is used (no licensing questions). The hand is drawn
// from 21 landmarks: a sphere per joint, a tapered cylinder per bone and a
// convex palm. Poses come from reference_poses.json, which is built from
// REAL recorded samples (scripts/app/build_reference_poses.py), not invented.
//
// Transitions keep bone lengths: each bone's direction is rotated smoothly
// (spherical interpolation) from one pose to the next.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";

// Parent of each landmark (MediaPipe hand topology). -1 = wrist (root).
const PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19];
const BONES = PARENT.map((p, i) => [p, i]).filter(([p]) => p >= 0)
  .concat([[5, 9], [9, 13], [13, 17]]);       // knuckle line
const PALM = [0, 1, 2, 5, 9, 13, 17];
// Joint radii in hand units (the hand is scaled to about 1.9 units tall).
const RADIUS = [0.13, 0.12, 0.105, 0.09, 0.08, 0.1, 0.088, 0.078, 0.07, 0.102, 0.09,
  0.08, 0.071, 0.098, 0.086, 0.076, 0.068, 0.09, 0.078, 0.068, 0.062];

// Landmarks (dataset coords: x right, y down, z = depth, negative is nearer
// the camera) -> Three.js coords (x right, y up, z towards the viewer).
function toVec(p) {
  return new THREE.Vector3(p[0], -p[1], -p[2]);
}

export class HandRig {
  constructor(canvas, { onError } = {}) {
    this.canvas = canvas;
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    } catch (e) {
      onError?.("3D view unavailable: WebGL could not start in this browser.");
      throw e;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
    // Slightly turned view (not straight-on) so finger depth is visible.
    this.camera.position.set(1.1, 0.55, 4.0).setLength(4.2);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.addEventListener("change", () => { this.dirty = true; });
    this.controls.minDistance = 2.5;
    this.controls.maxDistance = 8;
    this.controls.saveState();

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc8, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2, 3, 4);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xdfe8ff, 0.8);
    rim.position.set(-3, 1, -2);
    this.scene.add(rim);

    this.material = new THREE.MeshPhysicalMaterial({
      color: 0xd8d2cc, roughness: 0.5, metalness: 0.0, clearcoat: 0.25, clearcoatRoughness: 0.6,
    });
    this.root = new THREE.Group();        // whole-hand motion (J, Z, repeat slide)
    this.hand = new THREE.Group();        // centred hand
    this.root.add(this.hand);
    this.hand.visible = false;           // hidden until the first pose is set
    this.scene.add(this.root);

    const sphere = new THREE.SphereGeometry(1, 24, 16);
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 20, 1, true);
    this.joints = RADIUS.map((r) => {
      const m = new THREE.Mesh(sphere, this.material);
      m.scale.setScalar(r);
      this.hand.add(m);
      return m;
    });
    this.bones = BONES.map(([a, b]) => {
      const m = new THREE.Mesh(cyl, this.material);
      // knuckle-line bones are hidden inside the palm; finger bones are drawn
      m.userData = { a, b, r: Math.min(RADIUS[a], RADIUS[b]) * 0.97 };
      this.hand.add(m);
      return m;
    });
    this.palm = null;

    this.dirty = true;          // render only when something changed (saves CPU/GPU)
    this.points = null;         // current 21 Vector3
    this.anim = null;           // active pose transition
    this.motion = null;         // active whole-hand motion
    this._raf = 0;
    this._resize = new ResizeObserver(() => this.resize());
    this._resize.observe(canvas.parentElement);
    this.resize();
    this._loop = this._loop.bind(this);
    this._raf = requestAnimationFrame(this._loop);
  }

  resize() {
    const el = this.canvas.parentElement;
    const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // keep the whole hand visible in narrow panels
    this.camera.position.setLength(w / h < 0.8 ? 5.6 : 4.2);
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  resetView() {
    this.controls.reset();
    this.resize();
  }

  // ------------------------------------------------------------- poses
  static normalise(landmarks) {
    const pts = landmarks.map(toVec);
    const box = new THREE.Box3().setFromPoints(pts);
    const c = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const s = 1.9 / Math.max(size.x, size.y, 0.5);
    return pts.map((p) => p.sub(c).multiplyScalar(s));
  }

  setPose(landmarks, duration = 260) {
    this.live = false;
    const target = HandRig.normalise(landmarks);
    if (!this.points || duration <= 0) {
      this.points = target;
      this.anim = null;
      this._apply();
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.anim = { from: this.points.map((p) => p.clone()), to: target,
        start: performance.now(), duration, resolve };
    });
  }

  // Live hand: follow the camera's hand points (null = no hand, the hand is hidden).
  // Sized by the palm (wrist to middle knuckle), so it does not grow and shrink as the
  // fingers open and close, and smoothed a little for display only.
  // duration > 0: glide there over that many ms instead (used for the resting hand, which is set once).
  setLive(landmarks, { smoothing = 0.5, duration = 0 } = {}) {
    if (!landmarks) {
      this.live = false;
      if (this.hand.visible) { this.hand.visible = false; this.dirty = true; }
      return;
    }
    const pts = landmarks.map(toVec);
    const c = pts[9].clone();
    const s = 0.85 / Math.max(pts[0].distanceTo(pts[9]), 1e-6);
    const target = pts.map((p) => p.sub(c).multiplyScalar(s));
    if (duration > 0 && this.points) {
      this.live = true;
      this.anim = { from: this.points.map((p) => p.clone()), to: target, start: performance.now(), duration, resolve: () => {} };
      return;
    }
    this.anim = null;
    this.points = this.live && this.points ? this.points.map((p, i) => p.lerp(target[i], 1 - smoothing)) : target;
    this.live = true;
    this._apply();
  }

  // Whole-hand motion: "J", "Z" or "slide" (repeated letter cue).
  playMotion(kind, duration = 900) {
    return new Promise((resolve) => {
      this.motion = { kind, start: performance.now(), duration, resolve };
    });
  }

  stopMotion() {
    if (this.motion) this.motion.resolve();
    if (this.anim) this.anim.resolve();
    this.motion = null;
    if (this.anim) { this.points = this.anim.to; this.anim = null; this._apply(); }
    this.root.position.set(0, 0, 0);
    this.root.rotation.set(0, 0, 0);
    this.dirty = true;
  }

  _interpolate(from, to, t) {
    // Rebuild from the wrist outwards, slerping each bone direction.
    const out = new Array(21);
    out[0] = from[0].clone().lerp(to[0], t);
    for (let i = 1; i < 21; i++) {
      const p = PARENT[i];
      const a = from[i].clone().sub(from[p]);
      const b = to[i].clone().sub(to[p]);
      const la = a.length(), lb = b.length();
      const qa = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), a.normalize());
      const qb = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.normalize());
      const dir = new THREE.Vector3(0, 1, 0).applyQuaternion(qa.slerp(qb, t));
      out[i] = out[p].clone().add(dir.multiplyScalar(la + (lb - la) * t));
    }
    return out;
  }

  _apply() {
    const P = this.points;
    if (!P) return;
    this.hand.visible = true;
    this.dirty = true;
    this.joints.forEach((m, i) => m.position.copy(P[i]));
    const up = new THREE.Vector3(0, 1, 0);
    for (const m of this.bones) {
      const { a, b, r } = m.userData;
      const d = P[b].clone().sub(P[a]);
      const len = d.length();
      m.position.copy(P[a]).addScaledVector(d, 0.5);
      m.quaternion.setFromUnitVectors(up, d.normalize());
      m.scale.set(r, len, r);
    }
    // Palm: convex hull of the palm landmarks pushed slightly front and back.
    const n = new THREE.Vector3().crossVectors(P[5].clone().sub(P[0]), P[17].clone().sub(P[0]))
      .normalize().multiplyScalar(0.075);
    const hull = [];
    for (const i of PALM) hull.push(P[i].clone().add(n), P[i].clone().sub(n));
    try {
      let g = new ConvexGeometry(hull);
      g.deleteAttribute("normal");
      g = mergeVertices(g, 1e-4);      // share vertices -> smooth shading
      g.computeVertexNormals();
      if (this.palm) { this.palm.geometry.dispose(); this.palm.geometry = g; }
      else { this.palm = new THREE.Mesh(g, this.material); this.hand.add(this.palm); }
    } catch { /* degenerate hull for a moment: keep the previous palm */ }
  }

  _motionOffset(kind, s) {
    // Paths in hand units. s goes 0 -> 1.
    const ease = (x) => 0.5 - 0.5 * Math.cos(Math.PI * x);
    if (kind === "J") {
      // down, then hook toward the signer while the wrist turns
      const down = Math.min(1, s / 0.55);
      const hook = Math.max(0, (s - 0.55) / 0.45);
      return { x: -0.35 * ease(hook), y: -0.45 * ease(down) + 0.12 * ease(hook), rz: -0.9 * ease(s) };
    }
    if (kind === "Z") {
      const pts = [[-0.3, 0.3], [0.3, 0.3], [-0.3, -0.3], [0.3, -0.3]];
      const seg = Math.min(2, Math.floor(s * 3));
      const local = s * 3 - seg;
      const [x0, y0] = pts[seg], [x1, y1] = pts[seg + 1];
      return { x: x0 + (x1 - x0) * local, y: y0 + (y1 - y0) * local, rz: 0 };
    }
    // slide: a short sideways move and back (repeated letter cue)
    return { x: 0.28 * Math.sin(Math.PI * s), y: 0, rz: 0 };
  }

  _loop(now) {
    this._raf = requestAnimationFrame(this._loop);
    if (this.anim) {
      const t = Math.min(1, (now - this.anim.start) / this.anim.duration);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      this.points = this._interpolate(this.anim.from, this.anim.to, e);
      this._apply();
      if (t >= 1) { const r = this.anim.resolve; this.points = this.anim.to; this.anim = null; r(); }
    }
    if (this.motion) {
      const s = Math.min(1, (now - this.motion.start) / this.motion.duration);
      const o = this._motionOffset(this.motion.kind, s);
      this.root.position.set(o.x, o.y, 0);
      this.root.rotation.set(0, 0, o.rz);
      if (s >= 1) {
        const r = this.motion.resolve;
        this.motion = null;
        if (r) setTimeout(() => {
          this.root.position.set(0, 0, 0);
          this.root.rotation.set(0, 0, 0);
          this.dirty = true;
          r();
        }, 120);
      }
    }
    const moved = this.controls.update();      // true while the user rotates the view
    if (this.anim || this.motion || moved || this.dirty) {
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
    }
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._resize.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
