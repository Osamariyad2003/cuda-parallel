"use client"

import { useRef } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Text, OrbitControls, Environment } from "@react-three/drei"
import * as THREE from "three"
import SafeCanvasWrapper from "./safe-canvas-wrapper"
import FallbackCpuGpu from "./fallback-cpu-gpu"

function CPUModel() {
  const groupRef = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (!groupRef.current) return

    groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.1
  })

  return (
    <group ref={groupRef} position={[-2.5, 0, 0]}>
      {/* CPU Base */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[3, 0.2, 3]} />
        <meshStandardMaterial color="#444" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* CPU Cores - Few but larger */}
      {[
        [-0.8, 0.35, -0.8],
        [0.8, 0.35, -0.8],
        [-0.8, 0.35, 0.8],
        [0.8, 0.35, 0.8],
      ].map((position, index) => (
        <mesh key={index} position={position}>
          <boxGeometry args={[1, 0.5, 1]} />
          <meshStandardMaterial color="#777" metalness={0.9} roughness={0.1} />
        </mesh>
      ))}

      <Text position={[0, -1, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
        CPU
      </Text>

      <Text position={[0, -1.6, 0]} fontSize={0.3} color="#aaa" anchorX="center" anchorY="middle">
        Few Powerful Cores
      </Text>
    </group>
  )
}

function GPUModel() {
  const groupRef = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (!groupRef.current) return

    groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.1
  })

  return (
    <group ref={groupRef} position={[2.5, 0, 0]}>
      {/* GPU Base */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[3, 0.2, 3]} />
        <meshStandardMaterial color="#063" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* GPU Cores - Many small cores */}
      {Array.from({ length: 64 }).map((_, index) => {
        const row = Math.floor(index / 8)
        const col = index % 8
        const x = (col - 3.5) * 0.35
        const z = (row - 3.5) * 0.35

        return (
          <mesh key={index} position={[x, 0.25, z]}>
            <boxGeometry args={[0.3, 0.3, 0.3]} />
            <meshStandardMaterial
              color={new THREE.Color(0.1, 0.5 + (row / 8) * 0.5, 0.3 + (col / 8) * 0.3)}
              metalness={0.9}
              roughness={0.1}
            />
          </mesh>
        )
      })}

      <Text position={[0, -1, 0]} fontSize={0.5} color="white" anchorX="center" anchorY="middle">
        GPU
      </Text>

      <Text position={[0, -1.6, 0]} fontSize={0.3} color="#aaa" anchorX="center" anchorY="middle">
        Thousands of Cores
      </Text>
    </group>
  )
}

function CpuVsGpuCanvas() {
  return (
    <Canvas camera={{ position: [0, 3, 10], fov: 40 }}>
      <ambientLight intensity={0.5} />
      <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
      <CPUModel />
      <GPUModel />
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        makeDefault
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={Math.PI / 1.5}
      />
      <Environment preset="night" />
    </Canvas>
  )
}

export default function CpuVsGpu() {
  return (
    <SafeCanvasWrapper
      fallback={<FallbackCpuGpu />}
      className="w-full h-[400px] rounded-lg overflow-hidden border border-gray-700"
    >
      <CpuVsGpuCanvas />
    </SafeCanvasWrapper>
  )
}
