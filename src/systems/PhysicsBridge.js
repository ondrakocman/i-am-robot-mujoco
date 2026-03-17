/**
 * PhysicsBridge.js — Main-thread interface to the MuJoCo Web Worker
 *
 * Manages SharedArrayBuffer lifecycle, spawns the worker, and provides
 * read/write access to the physics state for the rendering thread.
 * Includes interpolation for smooth 90 Hz display from 60 Hz physics.
 */

// Must match layout in mujocoWorker.js
const OFFSET_TARGETS   = 0;
const OFFSET_QPOS      = 28;
const OFFSET_QVEL      = 56;
const OFFSET_TORQUES    = 84;
const OFFSET_OBJECTS    = 112;
const OFFSET_NCON       = 133;
const OFFSET_TIME       = 134;
const OFFSET_SEQ        = 135;
const NUM_ACTIVE_JOINTS = 28;
const NUM_OBJECTS        = 3;
const TOTAL_FLOATS       = 136;

// Joint names in actuator order (mirrors mujocoWorker.js)
export const ACTIVE_JOINT_NAMES = [
  // Left arm (7)
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
  'left_shoulder_yaw_joint', 'left_elbow_joint',
  'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  // Right arm (7)
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
  'right_shoulder_yaw_joint', 'right_elbow_joint',
  'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
  // Left hand (7)
  'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
  'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
  'left_hand_index_0_joint', 'left_hand_index_1_joint',
  // Right hand (7)
  'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
  'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
  'right_hand_index_0_joint', 'right_hand_index_1_joint',
];

export const OBJECT_NAMES = ['red_cube', 'blue_cube', 'green_cube'];

export default class PhysicsBridge {
  constructor() {
    this.worker = null;
    this.sab = null;
    this.f64 = null;
    this.i32Seq = null;
    this.ready = false;
    this._readyPromise = null;
    this._readyResolve = null;

    // Double-buffer for interpolation
    this._prevState = null;
    this._currState = null;
    this._lastSeq = -1;
    this._lastPhysicsTime = 0;
    this._lastReadTime = 0;
  }

  /**
   * Initialize the physics bridge: allocate SAB, spawn worker, load model.
   * @param {string} modelUrl - URL to the MJCF XML file (e.g. '/mujoco/g1_dex3_tabletop.xml')
   * @param {string[]} meshUrls - URLs to all mesh files needed by the model
   * @returns {Promise} resolves when worker reports ready
   */
  async init(modelUrl, meshUrls = []) {
    if (typeof SharedArrayBuffer === 'undefined') {
      throw new Error(
        'SharedArrayBuffer not available. Ensure COOP/COEP headers are set.'
      );
    }

    // Allocate SharedArrayBuffer (Float64 = 8 bytes each)
    this.sab = new SharedArrayBuffer(TOTAL_FLOATS * 8);
    this.f64 = new Float64Array(this.sab);
    this.i32Seq = new Int32Array(this.sab, OFFSET_SEQ * 8, 1);

    // Initialize interpolation buffers
    this._prevState = this._createStateSnapshot();
    this._currState = this._createStateSnapshot();

    // Create ready promise
    this._readyPromise = new Promise((resolve) => {
      this._readyResolve = resolve;
    });

    // Spawn worker
    this.worker = new Worker(
      new URL('./mujocoWorker.js', import.meta.url),
      { type: 'module' }
    );

    this.worker.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'ready') {
        this.ready = true;
        // Read initial state
        this._snapshotState();
        if (this._readyResolve) {
          this._readyResolve(e.data);
          this._readyResolve = null;
        }
      } else if (type === 'error') {
        console.error('[PhysicsBridge] Worker error:', e.data.message);
      } else if (type === 'reset_done') {
        this._snapshotState();
      }
    };

    this.worker.onerror = (e) => {
      console.error('[PhysicsBridge] Worker uncaught error:', e);
    };

    // Send init message
    this.worker.postMessage({
      type: 'init',
      modelUrl,
      meshUrls,
      sab: this.sab,
    });

    return this._readyPromise;
  }

  /**
   * Write target joint angles to the shared buffer.
   * @param {Float64Array|number[]} targets - 28 target angles
   *   [0..6] left arm, [7..13] right arm, [14..20] left hand, [21..27] right hand
   */
  writeTargets(targets) {
    if (!this.f64) return;
    for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
      this.f64[OFFSET_TARGETS + i] = targets[i] || 0;
    }
  }

  /**
   * Read the latest physics state from SAB.
   * Call this once per render frame before getInterpolatedState().
   */
  readState() {
    if (!this.f64) return null;

    const seq = Atomics.load(this.i32Seq, 0);
    if (seq !== this._lastSeq) {
      this._snapshotState();
      this._lastSeq = seq;
    }

    return this._currState;
  }

  /**
   * Get interpolated state between prev and current physics frames.
   * @param {number} alpha - Interpolation factor [0, 1]
   * @returns {object} { qpos[], qvel[], torques[], objects[], ncon, time }
   */
  getInterpolatedState(alpha = 1.0) {
    if (!this._currState) return null;

    const a = Math.max(0, Math.min(1, alpha));
    const prev = this._prevState;
    const curr = this._currState;

    const result = {
      qpos: new Float64Array(NUM_ACTIVE_JOINTS),
      qvel: new Float64Array(NUM_ACTIVE_JOINTS),
      torques: new Float64Array(NUM_ACTIVE_JOINTS),
      objects: new Array(NUM_OBJECTS),
      ncon: curr.ncon,
      time: curr.time,
    };

    // Lerp joint positions
    for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
      result.qpos[i] = prev.qpos[i] + a * (curr.qpos[i] - prev.qpos[i]);
      result.qvel[i] = curr.qvel[i];
      result.torques[i] = curr.torques[i];
    }

    // Lerp object positions, slerp quaternions
    for (let o = 0; o < NUM_OBJECTS; o++) {
      const p = prev.objects[o];
      const c = curr.objects[o];
      if (!p || !c) {
        result.objects[o] = c || p || { pos: [0,0,0], quat: [1,0,0,0] };
        continue;
      }
      result.objects[o] = {
        pos: [
          p.pos[0] + a * (c.pos[0] - p.pos[0]),
          p.pos[1] + a * (c.pos[1] - p.pos[1]),
          p.pos[2] + a * (c.pos[2] - p.pos[2]),
        ],
        quat: slerpQuat(p.quat, c.quat, a),
      };
    }

    return result;
  }

  /**
   * Get raw (non-interpolated) current state snapshot.
   */
  getRawState() {
    return this._currState;
  }

  /**
   * Reset the simulation to initial state.
   */
  reset() {
    if (this.worker) {
      this.worker.postMessage({ type: 'reset' });
    }
  }

  /**
   * Stop the physics loop and terminate the worker.
   */
  dispose() {
    if (this.worker) {
      this.worker.postMessage({ type: 'stop' });
      this.worker.terminate();
      this.worker = null;
    }
    this.ready = false;
    this.sab = null;
    this.f64 = null;
    this.i32Seq = null;
  }

  // ─── Private helpers ───

  _createStateSnapshot() {
    return {
      qpos: new Float64Array(NUM_ACTIVE_JOINTS),
      qvel: new Float64Array(NUM_ACTIVE_JOINTS),
      torques: new Float64Array(NUM_ACTIVE_JOINTS),
      objects: new Array(NUM_OBJECTS).fill(null).map(() => ({
        pos: [0, 0, 0],
        quat: [1, 0, 0, 0],
      })),
      ncon: 0,
      time: 0,
    };
  }

  _snapshotState() {
    // Rotate buffers
    const tmp = this._prevState;
    this._prevState = this._currState;
    this._currState = tmp;

    const f = this.f64;

    // Read qpos
    for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
      this._currState.qpos[i] = f[OFFSET_QPOS + i];
    }

    // Read qvel
    for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
      this._currState.qvel[i] = f[OFFSET_QVEL + i];
    }

    // Read torques
    for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
      this._currState.torques[i] = f[OFFSET_TORQUES + i];
    }

    // Read object poses
    for (let o = 0; o < NUM_OBJECTS; o++) {
      const base = OFFSET_OBJECTS + o * 7;
      this._currState.objects[o] = {
        pos: [f[base], f[base + 1], f[base + 2]],
        quat: [f[base + 3], f[base + 4], f[base + 5], f[base + 6]],
      };
    }

    this._currState.ncon = f[OFFSET_NCON];
    this._currState.time = f[OFFSET_TIME];
    this._lastReadTime = performance.now();
  }
}

// ─── Quaternion SLERP utility ───
function slerpQuat(a, b, t) {
  let ax = a[1], ay = a[2], az = a[3], aw = a[0];
  let bx = b[1], by = b[2], bz = b[3], bw = b[0];

  let dot = aw * bw + ax * bx + ay * by + az * bz;
  if (dot < 0) {
    bw = -bw; bx = -bx; by = -by; bz = -bz;
    dot = -dot;
  }

  if (dot > 0.9995) {
    // Linear interpolation for very close quaternions
    return [
      aw + t * (bw - aw),
      ax + t * (bx - ax),
      ay + t * (by - ay),
      az + t * (bz - az),
    ];
  }

  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;

  return [
    wa * aw + wb * bw,
    wa * ax + wb * bx,
    wa * ay + wb * by,
    wa * az + wb * bz,
  ];
}

export { NUM_ACTIVE_JOINTS, NUM_OBJECTS, OBJECT_NAMES as OBJECT_BODY_NAMES };
