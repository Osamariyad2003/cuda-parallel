"use client"

import { useRef, useMemo } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls, Environment } from "@react-three/drei"
import * as THREE from "three"
import SafeCanvasWrapper from "./safe-canvas-wrapper"
import FallbackCudaHero from "./fallback-cuda-hero"

function CudaGrid() {
  const groupRef = useRef<THREE.Group>(null)
  const cubeSize = 0.4
  const gap = 0.2
  const gridSize = 8
  const totalSize = gridSize * (cubeSize + gap) - gap
  const offset = -totalSize / 2 + cubeSize / 2

  // Create animation data for each cube
  const cubesData = useMemo(() => {
    return Array.from({ length: gridSize * gridSize * gridSize }).map((_, i) => {
      const x = Math.floor(i / (gridSize * gridSize))
      const y = Math.floor((i % (gridSize * gridSize)) / gridSize)
      const z = i % gridSize

      return {
        position: [offset + x * (cubeSize + gap), offset + y * (cubeSize + gap), offset + z * (cubeSize + gap)],
        color: new THREE.Color(0.1 + (x / gridSize) * 0.3, 0.5 + (y / gridSize) * 0.5, 0.3 + (z / gridSize) * 0.3),
        speed: 0.2 + Math.random() * 0.8,
        phase: Math.random() * Math.PI * 2,
      }
    })
  }, [])

  useFrame((state) => {
    if (!groupRef.current) return

    groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.1

    // Update each cube's position for wave effect
    groupRef.current.children.forEach((child, i) => {
      if (i >= cubesData.length) return

      const data = cubesData[i]
      const time = state.clock.getElapsedTime()

      // Apply a wave effect
      const wave = Math.sin(time * data.speed + data.phase) * 0.1
      child.position.y = data.position[1] + wave

      // Pulse effect on scale
      const pulse = 1 + Math.sin(time * data.speed * 0.5 + data.phase) * 0.05
      child.scale.set(pulse, pulse, pulse)
    })
  })

  return (
    <group ref={groupRef}>
      {cubesData.map((data, i) => (
        <mesh key={i} position={data.position as [number, number, number]}>
          <boxGeometry args={[cubeSize, cubeSize, cubeSize]} />
          <meshStandardMaterial
            color={data.color}
            metalness={0.8}
            roughness={0.2}
            emissive={data.color}
            emissiveIntensity={0.2}
          />
        </mesh>
      ))}
    </group>
  )
}

function CudaHeroCanvas() {
  return (
    <Canvas camera={{ position: [15, 10, 15], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
      <CudaGrid />
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
        makeDefault
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={Math.PI / 1.5}
      />
      <Environment preset="night" />
    </Canvas>
  )
}

export default function CudaHero() {
  return (
    <SafeCanvasWrapper
      fallback={<FallbackCudaHero />}
      className="absolute inset-0 bg-gradient-to-b from-black to-gray-900"
    >
      <CudaHeroCanvas />
    </SafeCanvasWrapper>
  )
}
