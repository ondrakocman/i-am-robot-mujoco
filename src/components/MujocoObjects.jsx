/**
 * MujocoObjects.jsx — Renders manipulable objects whose poses come from MuJoCo.
 *
 * Reads object positions and quaternions from PhysicsBridge state each frame
 * and applies them to plain Three.js meshes (no Rapier RigidBody wrappers).
 */

import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { OBJECT_NAMES } from '../systems/PhysicsBridge.js'

const TABLE_POS = [0, 0.78, 0]
const TABLE_SIZE = [0.5, 0.015, 0.3]
const TABLE_LEG_H = 0.39
const TABLE_LEG_R = 0.02

const LEG_OFFSETS = [
  [-0.46, -0.26],
  [ 0.46, -0.26],
  [-0.46,  0.26],
  [ 0.46,  0.26],
]

const OBJECT_COLORS = ['#cc3333', '#3366cc', '#33aa55']
const CUBE_SIZE = 0.025

const _quat = new THREE.Quaternion()

export function MujocoObjects({ physicsBridge }) {
  const cubeRefs = useRef([])

  useFrame(() => {
    if (!physicsBridge || !physicsBridge.ready) return

    const state = physicsBridge.readState()
    if (!state) return

    for (let i = 0; i < state.objects.length; i++) {
      const mesh = cubeRefs.current[i]
      if (!mesh) continue

      const obj = state.objects[i]
      if (!obj) continue

      mesh.position.set(obj.pos[0], obj.pos[2], -obj.pos[1])
      // MuJoCo quat: [w, x, y, z] → Three.js: swap Y/Z for coordinate system
      _quat.set(obj.quat[1], obj.quat[3], -obj.quat[2], obj.quat[0])
      mesh.quaternion.copy(_quat)
    }
  })

  return (
    <>
      {/* Static table */}
      <group>
        <mesh position={[TABLE_POS[0], TABLE_POS[1], TABLE_POS[2]]}>
          <boxGeometry args={[TABLE_SIZE[0] * 2, TABLE_SIZE[1] * 2, TABLE_SIZE[2] * 2]} />
          <meshStandardMaterial color="#8B6914" roughness={0.7} />
        </mesh>
        {LEG_OFFSETS.map(([dx, dz], i) => (
          <mesh key={i} position={[TABLE_POS[0] + dx, TABLE_LEG_H, TABLE_POS[2] + dz]}>
            <boxGeometry args={[TABLE_LEG_R * 2, TABLE_LEG_H * 2, TABLE_LEG_R * 2]} />
            <meshStandardMaterial color="#6B4914" roughness={0.8} />
          </mesh>
        ))}
      </group>

      {/* Dynamic cubes — poses set by MuJoCo physics */}
      {OBJECT_COLORS.map((color, i) => (
        <mesh
          key={OBJECT_NAMES[i]}
          ref={(el) => { cubeRefs.current[i] = el }}
        >
          <boxGeometry args={[CUBE_SIZE * 2, CUBE_SIZE * 2, CUBE_SIZE * 2]} />
          <meshStandardMaterial color={color} roughness={0.3} metalness={0.1} />
        </mesh>
      ))}
    </>
  )
}
