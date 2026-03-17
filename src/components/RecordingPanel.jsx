/**
 * RecordingPanel.jsx — VR-compatible recording controls
 *
 * Floating panel that follows the user's view in VR, providing
 * record/stop/export controls for the EpisodeRecorder.
 * Uses drei's Billboard for VR-friendly always-facing text.
 */

import { useRef, useState, useEffect, useCallback } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import EpisodeRecorder from '../systems/EpisodeRecorder.js'

const PANEL_OFFSET = new THREE.Vector3(0, -0.15, -0.45)
const _panelPos = new THREE.Vector3()

export function RecordingPanel({ physicsBridge, targetAnglesRef }) {
  const { camera } = useThree()
  const panelRef = useRef()
  const recorderRef = useRef(null)
  const [recording, setRecording] = useState(false)
  const [stats, setStats] = useState({ frameCount: 0, duration: 0, episodeCount: 0 })

  // Initialize recorder
  useEffect(() => {
    const recorder = new EpisodeRecorder()
    recorderRef.current = recorder
    return () => {
      recorder.dispose()
      recorderRef.current = null
    }
  }, [])

  // Toggle recording
  const toggleRecording = useCallback(() => {
    const recorder = recorderRef.current
    if (!recorder) return

    if (recorder.recording) {
      recorder.stopRecording().then(() => {
        setRecording(false)
        setStats(recorder.getStats())
      })
    } else if (physicsBridge && physicsBridge.ready) {
      recorder.startRecording(physicsBridge, targetAnglesRef?.current)
      setRecording(true)
    }
  }, [physicsBridge, targetAnglesRef])

  // Export all episodes
  const exportAll = useCallback(() => {
    recorderRef.current?.exportAllEpisodes()
  }, [])

  // Reset simulation
  const resetSim = useCallback(() => {
    if (physicsBridge) physicsBridge.reset()
  }, [physicsBridge])

  // Update stats display at ~4 Hz
  useFrame(() => {
    if (!recorderRef.current) return
    const now = performance.now()
    if (!panelRef.current) return

    // Position panel in front of camera
    _panelPos.copy(PANEL_OFFSET).applyQuaternion(camera.quaternion).add(camera.position)
    panelRef.current.position.lerp(_panelPos, 0.1)
    panelRef.current.quaternion.slerp(camera.quaternion, 0.1)

    // Update stats periodically
    if (recorderRef.current.recording) {
      const s = recorderRef.current.getStats()
      // Only re-render state every ~250ms to avoid excessive React updates
      if (Math.floor(now / 250) !== Math.floor((now - 16) / 250)) {
        setStats(s)
      }
    }
  })

  const bridgeReady = physicsBridge && physicsBridge.ready
  const statusColor = recording ? '#ff4444' : bridgeReady ? '#44ff44' : '#ffaa00'
  const statusText = recording
    ? `REC ${stats.duration.toFixed(1)}s (${stats.frameCount}f)`
    : bridgeReady
      ? `Ready | ${stats.episodeCount} eps`
      : 'Loading MuJoCo...'

  return (
    <group ref={panelRef}>
      {/* Background panel */}
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[0.24, 0.10]} />
        <meshBasicMaterial color="#1a1a2e" transparent opacity={0.85} side={THREE.DoubleSide} />
      </mesh>

      {/* Status indicator dot */}
      <mesh position={[-0.095, 0.03, 0.001]}>
        <circleGeometry args={[0.005, 16]} />
        <meshBasicMaterial color={statusColor} />
      </mesh>

      {/* Status text (sprite-based for VR readability) */}
      <StatusText text={statusText} position={[0.01, 0.03, 0.001]} fontSize={0.008} />

      {/* Record / Stop button */}
      <VRButton
        position={[-0.06, -0.02, 0.001]}
        size={[0.07, 0.03]}
        color={recording ? '#cc2222' : '#22aa44'}
        label={recording ? 'STOP' : 'REC'}
        onClick={toggleRecording}
      />

      {/* Export button */}
      <VRButton
        position={[0.02, -0.02, 0.001]}
        size={[0.07, 0.03]}
        color="#2266aa"
        label="EXPORT"
        onClick={exportAll}
      />

      {/* Reset button */}
      <VRButton
        position={[0.095, -0.02, 0.001]}
        size={[0.04, 0.03]}
        color="#666666"
        label="RST"
        onClick={resetSim}
      />
    </group>
  )
}

// Simple text using a canvas texture for VR rendering
function StatusText({ text, position, fontSize = 0.01 }) {
  const meshRef = useRef()
  const textureRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 512
    canvas.height = 64
    canvasRef.current = canvas
    const tex = new THREE.CanvasTexture(canvas)
    tex.minFilter = THREE.LinearFilter
    textureRef.current = tex
  }, [])

  useEffect(() => {
    if (!canvasRef.current || !textureRef.current) return
    const ctx = canvasRef.current.getContext('2d')
    ctx.clearRect(0, 0, 512, 64)
    ctx.fillStyle = '#ffffff'
    ctx.font = '28px monospace'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, 4, 32)
    textureRef.current.needsUpdate = true
  }, [text])

  if (!textureRef.current) return null

  return (
    <mesh ref={meshRef} position={position}>
      <planeGeometry args={[0.16, 0.02]} />
      <meshBasicMaterial map={textureRef.current} transparent />
    </mesh>
  )
}

// VR-interactive button (ray-click target)
function VRButton({ position, size, color, label, onClick }) {
  const meshRef = useRef()
  const textureRef = useRef(null)
  const [hovered, setHovered] = useState(false)

  useEffect(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 256
    canvas.height = 64
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = color
    ctx.roundRect(0, 0, 256, 64, 8)
    ctx.fill()
    ctx.fillStyle = '#ffffff'
    ctx.font = 'bold 28px monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, 128, 32)
    const tex = new THREE.CanvasTexture(canvas)
    tex.minFilter = THREE.LinearFilter
    textureRef.current = tex
  }, [color, label])

  if (!textureRef.current) return null

  return (
    <mesh
      ref={meshRef}
      position={position}
      onClick={(e) => { e.stopPropagation(); onClick?.() }}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      scale={hovered ? 1.1 : 1}
    >
      <planeGeometry args={size} />
      <meshBasicMaterial map={textureRef.current} transparent />
    </mesh>
  )
}
