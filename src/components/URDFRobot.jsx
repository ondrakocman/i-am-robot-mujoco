import { useRef, useEffect, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import URDFLoader from 'urdf-loader'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { solveCCDIK } from '../systems/CCDIK.js'
import { retargetHand, RetargetingFilter } from '../systems/HandRetargeting.js'
import { ExponentialSmoother, QuaternionSmoother } from '../systems/ImpedanceControl.js'
import { WeightedMovingFilter } from '../systems/WeightedMovingFilter.js'
import { ACTIVE_JOINT_NAMES } from '../systems/PhysicsBridge.js'
import { XR_JOINT_NAMES } from '../constants/kinematics.js'

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }

const MAT_BODY = new THREE.MeshStandardMaterial({ color: 0x4a4a6e, roughness: 0.4, metalness: 0.25 })
const MAT_ACCENT = new THREE.MeshStandardMaterial({ color: 0x6a6a9e, roughness: 0.35, metalness: 0.3 })

const ROBOT_BASE_QUAT = new THREE.Quaternion()
;(() => {
  const qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
  const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2)
  ROBOT_BASE_QUAT.multiplyQuaternions(qy, qx)
})()

const ARM_CHAIN = {
  left: [
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  ],
  right: [
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
  ],
}
const HAND_LINK = { left: 'left_hand_palm_link', right: 'right_hand_palm_link' }

const EYE_LINK = 'mid360_link'
const EYE_LINK_FALLBACK = 'head_link'

// Frame correction: WebXR wrist has -Z=fingers, +Y=back-of-hand.
// URDF palm has +X=fingers. Left palm faces -Y, right palm faces +Y
// (confirmed by mirrored finger curl limits in the URDF).
// Left:  Ry(π/2) aligns fingers (-Z→+X) and keeps Y axis.
// Right: Rz(π)·Ry(π/2) also flips the palm normal axis.
const XR_TO_URDF_L = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(0, 1, 0), Math.PI / 2
)
const XR_TO_URDF_R = new THREE.Quaternion().multiplyQuaternions(
  new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI),
  new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2),
)

const _eyeWorld = new THREE.Vector3()
const _wristPos = new THREE.Vector3()
const _wristQuat = new THREE.Quaternion()
const _correctedQuat = new THREE.Quaternion()


export function URDFRobot({ vrMode = 'unlocked', worldRef, physicsBridge, onTargetAnglesRef }) {
  const { gl, camera } = useThree()
  const groupRef = useRef()
  const [robot, setRobot] = useState(null)
  const calibrated = useRef(false)
  const modeRef = useRef(vrMode)
  modeRef.current = vrMode

  const smoothL = useRef({ pos: new ExponentialSmoother(0.3), quat: new QuaternionSmoother(0.3) })
  const smoothR = useRef({ pos: new ExponentialSmoother(0.3), quat: new QuaternionSmoother(0.3) })
  const jointFilterL = useRef(new WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], 7))
  const jointFilterR = useRef(new WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], 7))
  const retargetL = useRef(new RetargetingFilter(0.4))
  const retargetR = useRef(new RetargetingFilter(0.4))

  // Buffer for 28 target joint angles sent to MuJoCo each frame
  const targetAngles = useRef(new Float64Array(28))

  // Expose targetAngles ref to parent for EpisodeRecorder
  useEffect(() => {
    if (onTargetAnglesRef) onTargetAnglesRef(targetAngles)
  }, [onTargetAnglesRef])

  // ── Load URDF ────────────────────────────────────────────────────────────

  useEffect(() => {
    const loader = new URDFLoader()
    const stlLoader = new STLLoader()
    const base = import.meta.env.BASE_URL + 'models/'
    let pending = 0
    let parsed = null
    let cancelled = false

    const tryFinalize = () => {
      if (pending === 0 && parsed && !cancelled) setRobot(parsed)
    }

    loader.loadMeshCb = (path, _mgr, onLoad) => {
      pending++
      stlLoader.load(base + path, (geo) => {
        geo.computeVertexNormals()
        const accent = /contour|shoulder_roll|shoulder_pitch|waist|logo/.test(path)
        const mesh = new THREE.Mesh(geo, (accent ? MAT_ACCENT : MAT_BODY).clone())
        const g = new THREE.Group()
        g.add(mesh)
        onLoad(g)
        pending--
        tryFinalize()
      }, undefined, () => { onLoad(new THREE.Group()); pending--; tryFinalize() })
    }

    fetch(base + 'g1.urdf')
      .then(r => r.text())
      .then(text => { parsed = loader.parse(text); tryFinalize() })
      .catch(e => console.error('URDF load failed:', e))

    return () => { cancelled = true }
  }, [])

  // ── Setup robot ────────────────────────────────────────────────────────

  useEffect(() => {
    if (!robot || !groupRef.current) return

    robot.quaternion.copy(ROBOT_BASE_QUAT)
    groupRef.current.add(robot)
    groupRef.current.position.set(0, 0, 0)

    if (worldRef?.current) {
      worldRef.current.position.set(0, 0, 0)
      worldRef.current.rotation.set(0, 0, 0)
    }

    groupRef.current.position.y = 0.75
    groupRef.current.updateMatrixWorld(true)

    calibrated.current = false

    return () => {
      groupRef.current?.remove(robot)
    }
  }, [robot])

  useEffect(() => {
    if (!robot?.links?.head_link) return
    robot.links.head_link.visible = vrMode !== 'locked'
  }, [robot, vrMode])

  // ── Main frame loop ──────────────────────────────────────────────────────

  useFrame((_state, _delta, xrFrame) => {
    if (!robot || !groupRef.current) return

    const eyeLink = robot.links?.[EYE_LINK] || robot.links?.[EYE_LINK_FALLBACK]

    if (xrFrame && eyeLink) {
      if (!calibrated.current) {
        calibrateView(modeRef.current, camera, groupRef, worldRef, eyeLink)
        calibrated.current = true
      } else if (modeRef.current === 'locked' && worldRef?.current) {
        lockViewToEye(camera, worldRef, eyeLink)
      }
    }

    if (!xrFrame) return
    const session = gl.xr.getSession()
    const refSpace = gl.xr.getReferenceSpace()
    if (!session || !refSpace) return

    for (const source of session.inputSources) {
      if (!source.hand) continue
      const side = source.handedness
      if (side !== 'left' && side !== 'right') continue

      const xrJoints = readXRJoints(xrFrame, source, refSpace)
      if (!xrJoints['wrist']) continue

      const correction = side === 'left' ? XR_TO_URDF_L : XR_TO_URDF_R
      _correctedQuat.copy(xrJoints['wrist'].quaternion).multiply(correction)

      const sm = side === 'left' ? smoothL.current : smoothR.current
      _wristPos.copy(sm.pos.update(xrJoints['wrist'].position))
      _wristQuat.copy(sm.quat.update(_correctedQuat))

      const chain = ARM_CHAIN[side].map(n => robot.joints?.[n]).filter(Boolean)
      const endLink = robot.links?.[HAND_LINK[side]]

      if (chain.length > 0 && endLink) {
        const filter = side === 'left' ? jointFilterL.current : jointFilterR.current
        solveAndFilter(chain, endLink, _wristPos, _wristQuat, filter)
      }

      // Retarget hand fingers
      const raw = retargetHand(xrJoints)
      const rt = side === 'left' ? retargetL.current : retargetR.current
      applyFingerAngles(robot, side, rt.update(raw))
    }

    // ── MuJoCo bridge: send targets and apply solved state ────────────
    if (physicsBridge && physicsBridge.ready) {
      // Collect current joint angles into the 28-element target buffer
      const tgt = targetAngles.current
      for (let i = 0; i < ACTIVE_JOINT_NAMES.length; i++) {
        const j = robot.joints?.[ACTIVE_JOINT_NAMES[i]]
        tgt[i] = j ? (j.angle || 0) : 0
      }
      physicsBridge.writeTargets(tgt)

      // Read solved state from MuJoCo and apply to visual model
      const state = physicsBridge.readState()
      if (state) {
        for (let i = 0; i < ACTIVE_JOINT_NAMES.length; i++) {
          const j = robot.joints?.[ACTIVE_JOINT_NAMES[i]]
          if (j?.setJointValue) j.setJointValue(state.qpos[i])
        }
      }
    }
  })

  return <group ref={groupRef} />
}

// ── Helpers ────────────────────────────────────────────────────────────────

function calibrateView(mode, camera, groupRef, worldRef, eyeLink) {
  const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')

  if (mode === 'locked' && worldRef?.current) {
    worldRef.current.rotation.y = euler.y
    worldRef.current.updateMatrixWorld(true)
    eyeLink.getWorldPosition(_eyeWorld)
    worldRef.current.position.x += camera.position.x - _eyeWorld.x
    worldRef.current.position.y += camera.position.y - _eyeWorld.y
    worldRef.current.position.z += camera.position.z - _eyeWorld.z
  } else {
    const savedY = groupRef.current.position.y
    groupRef.current.rotation.y = euler.y
    groupRef.current.updateMatrixWorld(true)
    eyeLink.getWorldPosition(_eyeWorld)
    groupRef.current.position.x += camera.position.x - _eyeWorld.x
    groupRef.current.position.z += camera.position.z - _eyeWorld.z
    groupRef.current.position.y = savedY
  }
}

function lockViewToEye(camera, worldRef, eyeLink) {
  worldRef.current.updateMatrixWorld(true)
  eyeLink.getWorldPosition(_eyeWorld)
  worldRef.current.position.x += camera.position.x - _eyeWorld.x
  worldRef.current.position.z += camera.position.z - _eyeWorld.z
}

function readXRJoints(xrFrame, source, refSpace) {
  const joints = {}
  for (const name of XR_JOINT_NAMES) {
    const space = source.hand.get(name)
    if (!space) continue
    const pose = xrFrame.getJointPose(space, refSpace)
    if (!pose) continue
    const p = pose.transform.position
    const o = pose.transform.orientation
    joints[name] = {
      position: new THREE.Vector3(p.x, p.y, p.z),
      quaternion: new THREE.Quaternion(o.x, o.y, o.z, o.w),
    }
  }
  return joints
}

function solveAndFilter(chain, endLink, targetPos, targetQuat, filter) {
  solveCCDIK(chain, endLink, targetPos, targetQuat, 20)
  const solved = chain.map(j => j.angle || 0)
  const filtered = filter.addData(solved)
  chain.forEach((j, i) => {
    if (j.setJointValue && i < filtered.length) j.setJointValue(filtered[i])
  })
}

function curlToAngle(joint, curl) {
  if (!joint?.limit) return 0
  const { lower, upper } = joint.limit
  return Math.abs(lower) > Math.abs(upper) ? lower * curl : upper * curl
}

function applyFingerAngles(robot, side, data) {
  if (!robot.joints || !data) return
  const p = side + '_hand_'

  const t0 = robot.joints[p + 'thumb_0_joint']
  if (t0?.limit) {
    const range = Math.min(Math.abs(t0.limit.lower), Math.abs(t0.limit.upper))
    t0.setJointValue(clamp(data.thumb.abduction * range, t0.limit.lower, t0.limit.upper))
  }

  const t1 = robot.joints[p + 'thumb_1_joint']
  const t2 = robot.joints[p + 'thumb_2_joint']
  if (t1) t1.setJointValue(curlToAngle(t1, data.thumb.curl[0]))
  if (t2) t2.setJointValue(curlToAngle(t2, data.thumb.curl[1]))

  for (const finger of ['index', 'middle']) {
    const j0 = robot.joints[p + finger + '_0_joint']
    const j1 = robot.joints[p + finger + '_1_joint']
    if (j0) j0.setJointValue(curlToAngle(j0, data[finger].curl[0]))
    if (j1) j1.setJointValue(curlToAngle(j1, data[finger].curl[1]))
  }
}

// ── Tracking HUD ──────────────────────────────────────────────────────────

export function TrackingHUD() {
  const ref = useRef()
  const { gl } = useThree()

  useFrame((_, __, xrFrame) => {
    if (!xrFrame || !ref.current) return
    const session = gl.xr.getSession()
    if (!session) return
    let l = false, r = false
    for (const src of session.inputSources) {
      if (src.hand) {
        if (src.handedness === 'left') l = true
        if (src.handedness === 'right') r = true
      }
    }
    ref.current.material?.color.set(l && r ? '#00ff88' : l || r ? '#ffaa00' : '#ff4444')
  })

  return (
    <mesh ref={ref} position={[0.28, -0.25, -0.5]}>
      <sphereGeometry args={[0.005, 6, 6]} />
      <meshBasicMaterial color="#ff4444" />
    </mesh>
  )
}
