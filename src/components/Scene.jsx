import { useRef, useEffect, useState, useCallback } from 'react'
import { URDFRobot, TrackingHUD } from './URDFRobot.jsx'
import { MujocoObjects } from './MujocoObjects.jsx'
import { HandDebugPoints } from './HandDebugPoints.jsx'
import { RecordingPanel } from './RecordingPanel.jsx'
import PhysicsBridge from '../systems/PhysicsBridge.js'

function Environment() {
  return (
    <>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[30, 30]} />
        <meshStandardMaterial color="#3a4a5a" roughness={0.9} metalness={0.0} />
      </mesh>

      <gridHelper args={[30, 60, '#5588aa', '#445566']} position={[0, 0.001, 0]} />

      <directionalLight position={[5, 10, 7]} intensity={4} color="#ffffff" />
      <directionalLight position={[-4, 6, -3]} intensity={2} color="#ffffff" />
      <directionalLight position={[0, 4, 8]} intensity={2} color="#eeeeff" />
      <ambientLight intensity={2.0} color="#ffffff" />
      <hemisphereLight skyColor="#aaccee" groundColor="#555555" intensity={1.5} />
    </>
  )
}

// Collect all mesh URLs from the public/mujoco/meshes directory
function buildMeshUrls(base) {
  // All STL mesh files referenced by the MJCF model
  const meshFiles = [
    'pelvis.STL', 'pelvis_contour_link.STL',
    'waist_yaw_link.STL', 'waist_roll_link.STL', 'torso_link.STL',
    'logo_link.STL', 'head_link.STL', 'waist_support_link.STL',
    'left_shoulder_pitch_link.STL', 'left_shoulder_roll_link.STL',
    'left_shoulder_yaw_link.STL', 'left_elbow_link.STL',
    'left_wrist_roll_link.STL', 'left_wrist_pitch_link.STL', 'left_wrist_yaw_link.STL',
    'right_shoulder_pitch_link.STL', 'right_shoulder_roll_link.STL',
    'right_shoulder_yaw_link.STL', 'right_elbow_link.STL',
    'right_wrist_roll_link.STL', 'right_wrist_pitch_link.STL', 'right_wrist_yaw_link.STL',
    'left_hip_pitch_link.STL', 'left_hip_roll_link.STL', 'left_hip_yaw_link.STL',
    'left_knee_link.STL', 'left_ankle_pitch_link.STL', 'left_ankle_roll_link.STL',
    'right_hip_pitch_link.STL', 'right_hip_roll_link.STL', 'right_hip_yaw_link.STL',
    'right_knee_link.STL', 'right_ankle_pitch_link.STL', 'right_ankle_roll_link.STL',
    'left_hand_palm_link.STL', 'left_hand_thumb_0_link.STL', 'left_hand_thumb_1_link.STL',
    'left_hand_thumb_2_link.STL', 'left_hand_middle_0_link.STL', 'left_hand_middle_1_link.STL',
    'left_hand_index_0_link.STL', 'left_hand_index_1_link.STL',
    'right_hand_palm_link.STL', 'right_hand_thumb_0_link.STL', 'right_hand_thumb_1_link.STL',
    'right_hand_thumb_2_link.STL', 'right_hand_middle_0_link.STL', 'right_hand_middle_1_link.STL',
    'right_hand_index_0_link.STL', 'right_hand_index_1_link.STL',
  ]
  return meshFiles.map(f => `${base}/meshes/${f}`)
}

export function Scene({ vrMode }) {
  const worldRef = useRef()
  const [physicsBridge, setPhysicsBridge] = useState(null)
  const bridgeRef = useRef(null)
  const targetAnglesRef = useRef(null)

  const handleTargetAnglesRef = useCallback((ref) => {
    targetAnglesRef.current = ref
  }, [])

  useEffect(() => {
    const bridge = new PhysicsBridge()
    bridgeRef.current = bridge

    const base = import.meta.env.BASE_URL + 'mujoco'
    const modelUrl = `${base}/g1_dex3_tabletop.xml`
    const meshUrls = buildMeshUrls(base)

    bridge.init(modelUrl, meshUrls).then(() => {
      console.log('[Scene] MuJoCo PhysicsBridge ready')
      setPhysicsBridge(bridge)
    }).catch((err) => {
      console.error('[Scene] PhysicsBridge init failed:', err)
    })

    return () => {
      bridge.dispose()
      bridgeRef.current = null
    }
  }, [])

  return (
    <>
      <group ref={worldRef}>
        <Environment />
        <URDFRobot
          vrMode={vrMode}
          worldRef={worldRef}
          physicsBridge={physicsBridge}
          onTargetAnglesRef={handleTargetAnglesRef}
        />
        <MujocoObjects physicsBridge={physicsBridge} />
      </group>
      <RecordingPanel physicsBridge={physicsBridge} targetAnglesRef={targetAnglesRef} />
      <HandDebugPoints />
      <TrackingHUD />
    </>
  )
}
