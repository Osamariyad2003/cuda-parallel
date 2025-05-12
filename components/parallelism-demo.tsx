"use client"

import { useRef, useState, useEffect } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Text, OrbitControls, Environment, Html } from "@react-three/drei"
import * as THREE from "three"
import SafeCanvasWrapper from "./safe-canvas-wrapper"
import FallbackParallelism from "./fallback-parallelism"

function ThreadBlock({
  position,
  size,
  color,
  active,
  label,
  data,
}: {
  position: [number, number, number]
  size: number
  color: THREE.Color
  active: boolean
  label: string
  data: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)

  useFrame(() => {
    if (!meshRef.current) return

    if (active) {
      meshRef.current.scale.x = THREE.MathUtils.lerp(meshRef.current.scale.x, 1.2, 0.1)
      meshRef.current.scale.y = THREE.MathUtils.lerp(meshRef.current.scale.y, 1.2, 0.1)
      meshRef.current.scale.z = THREE.MathUtils.lerp(meshRef.current.scale.z, 1.2, 0.1)
    } else {
      meshRef.current.scale.x = THREE.MathUtils.lerp(meshRef.current.scale.x, 1, 0.1)
      meshRef.current.scale.y = THREE.MathUtils.lerp(meshRef.current.scale.y, 1, 0.1)
      meshRef.current.scale.z = THREE.MathUtils.lerp(meshRef.current.scale.z, 1, 0.1)
    }
  })

  return (
    <group position={position}>
      <mesh ref={meshRef} onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
        <boxGeometry args={[size, size, size]} />
        <meshStandardMaterial
          color={color}
          metalness={0.5}
          roughness={0.2}
          emissive={active ? color : new THREE.Color(0, 0, 0)}
          emissiveIntensity={active ? 0.5 : 0}
        />
      </mesh>
      <Text position={[0, -size / 1.5, 0]} fontSize={0.15} color="white" anchorX="center" anchorY="middle">
        {label}
      </Text>

      {hovered && (
        <Html position={[0, size, 0]} center>
          <div className="bg-black/80 text-white text-xs p-2 rounded pointer-events-none whitespace-nowrap">
            {active ? `Processing: ${data}` : "Idle"}
          </div>
        </Html>
      )}
    </group>
  )
}

function ParallelProcessingDemo() {
  const [activeBlocks, setActiveBlocks] = useState<number[]>([])
  const [blockData, setBlockData] = useState<number[]>(Array(16).fill(0))
  const groupRef = useRef<THREE.Group>(null)

  useEffect(() => {
    // Create a stable interval reference
    const intervalId = setInterval(() => {
      const newActiveBlocks: number[] = []
      const count = Math.floor(Math.random() * 8) + 4 // Activate 4-12 blocks at random

      // Generate new random data for each block
      const newBlockData = Array(16)
        .fill(0)
        .map(() => Math.floor(Math.random() * 100))

      for (let i = 0; i < count; i++) {
        const blockIndex = Math.floor(Math.random() * 16)
        if (!newActiveBlocks.includes(blockIndex)) {
          newActiveBlocks.push(blockIndex)
        }
      }

      setActiveBlocks(newActiveBlocks)
      setBlockData(newBlockData)
    }, 800)

    // Clean up interval on unmount
    return () => clearInterval(intervalId)
  }, []) // Empty dependency array ensures this only runs once

  useFrame((state) => {
    if (!groupRef.current) return

    groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.1
  })

  // Create a 4x4 grid of thread blocks
  const blocks = []
  const size = 0.6
  const gap = 0.3
  const gridWidth = 4
  const totalWidth = gridWidth * (size + gap) - gap
  const offset = -totalWidth / 2 + size / 2

  for (let x = 0; x < gridWidth; x++) {
    for (let z = 0; z < gridWidth; z++) {
      const index = x * gridWidth + z
      const isActive = activeBlocks.includes(index)

      // Different colors for different thread blocks
      const hue = (index / (gridWidth * gridWidth)) * 0.3 + 0.3 // Range from 0.3 to 0.6 (greens)
      const color = new THREE.Color().setHSL(hue, 0.8, 0.5)

      blocks.push(
        <ThreadBlock
          key={index}
          position={[offset + x * (size + gap), 0, offset + z * (size + gap)]}
          size={size}
          color={color}
          active={isActive}
          label={`Thread ${index}`}
          data={blockData[index]}
        />,
      )
    }
  }

  return (
    <group ref={groupRef}>
      {/* Grid base */}
      <mesh position={[0, -0.5, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[totalWidth + 1, totalWidth + 1]} />
        <meshStandardMaterial color="#111" />
      </mesh>

      {/* Thread blocks */}
      {blocks}

      {/* Information text */}
      <Text position={[0, -1.2, 0]} fontSize={0.25} color="white" anchorX="center" anchorY="middle" maxWidth={4}>
        CUDA Thread Blocks
      </Text>

      <Text position={[0, -1.6, 0]} fontSize={0.2} color="#aaa" anchorX="center" anchorY="middle" maxWidth={4}>
        {`${activeBlocks.length} threads active`}
      </Text>
    </group>
  )
}

function ParallelismDemoCanvas() {
  return (
    <Canvas camera={{ position: [6, 4, 6], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
      <ParallelProcessingDemo />
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

export default function ParallelismDemo() {
  return (
    <SafeCanvasWrapper
      fallback={<FallbackParallelism />}
      className="w-full h-[400px] rounded-lg overflow-hidden border border-gray-700"
    >
      <ParallelismDemoCanvas />
    </SafeCanvasWrapper>
  )
}
