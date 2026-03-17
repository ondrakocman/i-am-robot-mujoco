/**
 * mujocoWorker.js — MuJoCo WASM Physics Web Worker
 *
 * Runs the MuJoCo simulation at 60 Hz in a dedicated Web Worker thread.
 * Communicates with the main thread via SharedArrayBuffer for zero-copy
 * transfer of joint targets (main→worker) and solved state (worker→main).
 *
 * Messages:
 *   init  { modelUrl, meshUrls, sab }  → loads model, starts loop
 *   reset {}                            → resets simulation to initial state
 *   stop  {}                            → stops the simulation loop
 */

import MujocoModule from '@mujoco/mujoco';

// ─── Shared buffer layout (Float64) ───
// [0..27]     main→worker   target joint angles (28 active joints)
// [28..55]    worker→main   solved qpos
// [56..83]    worker→main   solved qvel
// [84..111]   worker→main   actuator forces
// [112..132]  worker→main   object poses (3 objects × 7 floats each = 21)
// [133]       worker→main   ncon (contact count)
// [134]       worker→main   simulation time
// [135]       sync flag     sequence counter (Int32 view at byte offset 135*8)
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
const PHYSICS_DT_MS      = 16; // ~60 Hz

// Joint names in actuator order (must match MJCF actuator order)
const ACTIVE_JOINT_NAMES = [
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

const OBJECT_BODY_NAMES = ['red_cube', 'blue_cube', 'green_cube'];

let mujoco = null;
let model = null;
let data = null;
let f64 = null;       // Float64Array view of SharedArrayBuffer
let i32Seq = null;    // Int32Array view for atomic sequence counter
let running = false;
let intervalId = null;

// Maps from joint name → index in qpos/qvel/ctrl arrays
let jointQposIdx = [];   // length NUM_ACTIVE_JOINTS
let jointQvelIdx = [];
let objectBodyIds = [];  // body ids for freejoint objects

async function init(msg) {
  const { modelUrl, meshUrls, sab } = msg;

  // Set up shared buffer views
  f64 = new Float64Array(sab);
  // Atomic sequence counter lives at byte offset OFFSET_SEQ * 8
  i32Seq = new Int32Array(sab, OFFSET_SEQ * 8, 1);

  // Load MuJoCo WASM
  mujoco = await MujocoModule();
  const FS = mujoco.FS;

  // Create directories in Emscripten FS
  FS.mkdir('/model');
  FS.mkdir('/model/meshes');

  // Fetch and write mesh files to Emscripten FS
  if (meshUrls && meshUrls.length > 0) {
    const meshPromises = meshUrls.map(async (url) => {
      const filename = url.split('/').pop();
      try {
        const resp = await fetch(url);
        const buf = await resp.arrayBuffer();
        FS.writeFile(`/model/meshes/${filename}`, new Uint8Array(buf));
      } catch (e) {
        console.warn(`[mujocoWorker] Failed to fetch mesh: ${url}`, e);
      }
    });
    await Promise.all(meshPromises);
  }

  // Fetch and write the model XML
  const xmlResp = await fetch(modelUrl);
  const xmlText = await xmlResp.text();
  FS.writeFile('/model/scene.xml', xmlText);

  // Load model from the Emscripten FS
  model = mujoco.MjModel.from_xml_path('/model/scene.xml');
  data = new mujoco.MjData(model, null);

  // Build joint index maps
  jointQposIdx = new Array(NUM_ACTIVE_JOINTS);
  jointQvelIdx = new Array(NUM_ACTIVE_JOINTS);
  for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
    const jntAccessor = model.jnt(ACTIVE_JOINT_NAMES[i]);
    jointQposIdx[i] = jntAccessor.qposadr;
    jointQvelIdx[i] = jntAccessor.dofadr;
  }

  // Build object body id list
  objectBodyIds = OBJECT_BODY_NAMES.map(name => {
    const bodyAccessor = model.body(name);
    return bodyAccessor.id;
  });

  // Initial forward kinematics
  mujoco.mj_forward(model, data);

  // Write initial state to SAB
  writeStateToBuf();

  self.postMessage({ type: 'ready', nq: model.nq, nv: model.nv, nu: model.nu });
}

function writeStateToBuf() {
  const qpos = data.qpos;
  const qvel = data.qvel;
  const actuatorForce = data.actuator_force;

  // Write solved qpos for active joints
  for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
    f64[OFFSET_QPOS + i] = qpos[jointQposIdx[i]];
  }

  // Write solved qvel for active joints
  for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
    f64[OFFSET_QVEL + i] = qvel[jointQvelIdx[i]];
  }

  // Write actuator forces
  for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
    f64[OFFSET_TORQUES + i] = actuatorForce[i];
  }

  // Write object poses (freejoint bodies: 7 floats = x,y,z, qw,qx,qy,qz)
  for (let o = 0; o < NUM_OBJECTS; o++) {
    const bodyId = objectBodyIds[o];
    const xpos = data.body(bodyId).xpos;
    const xquat = data.body(bodyId).xquat;
    const base = OFFSET_OBJECTS + o * 7;
    f64[base + 0] = xpos[0];
    f64[base + 1] = xpos[1];
    f64[base + 2] = xpos[2];
    f64[base + 3] = xquat[0]; // w
    f64[base + 4] = xquat[1]; // x
    f64[base + 5] = xquat[2]; // y
    f64[base + 6] = xquat[3]; // z
  }

  // Write contact count and time
  f64[OFFSET_NCON] = data.ncon;
  f64[OFFSET_TIME] = data.time;

  // Increment sequence counter atomically
  Atomics.add(i32Seq, 0, 1);
}

function step() {
  if (!model || !data) return;

  // Read target joint angles from SAB
  const ctrl = data.ctrl;
  for (let i = 0; i < NUM_ACTIVE_JOINTS; i++) {
    ctrl[i] = f64[OFFSET_TARGETS + i];
  }

  // Step physics
  mujoco.mj_step(model, data);

  // Write solved state back to SAB
  writeStateToBuf();
}

function startLoop() {
  if (running) return;
  running = true;
  intervalId = setInterval(step, PHYSICS_DT_MS);
}

function stopLoop() {
  running = false;
  if (intervalId !== null) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

function reset() {
  if (!model || !data) return;
  stopLoop();
  mujoco.mj_resetData(model, data);
  mujoco.mj_forward(model, data);
  writeStateToBuf();
  self.postMessage({ type: 'reset_done' });
}

// ─── Message handler ───
self.onmessage = async (e) => {
  const { type, ...payload } = e.data;
  try {
    switch (type) {
      case 'init':
        await init(payload);
        startLoop();
        break;
      case 'reset':
        reset();
        startLoop();
        break;
      case 'stop':
        stopLoop();
        break;
      default:
        console.warn(`[mujocoWorker] Unknown message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message, stack: err.stack });
  }
};
