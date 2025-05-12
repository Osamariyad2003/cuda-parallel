"use client"

import { useState, useEffect } from "react"
import { Zap } from "lucide-react"

export default function FallbackParallelism() {
  const [activeBlocks, setActiveBlocks] = useState<number[]>([])

  useEffect(() => {
    // Simulate parallel processing by activating blocks in waves
    const interval = setInterval(() => {
      const newActiveBlocks: number[] = []
      const count = Math.floor(Math.random() * 8) + 4 // Activate 4-12 blocks at random

      for (let i = 0; i < count; i++) {
        const blockIndex = Math.floor(Math.random() * 16)
        if (!newActiveBlocks.includes(blockIndex)) {
          newActiveBlocks.push(blockIndex)
        }
      }

      setActiveBlocks(newActiveBlocks)
    }, 800)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="w-full h-[400px] rounded-lg overflow-hidden border border-gray-700 bg-gray-900 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md">
        <h3 className="text-white font-medium text-center mb-6">CUDA Thread Blocks</h3>

        <div className="grid grid-cols-4 gap-4 mb-8">
          {Array.from({ length: 16 }).map((_, i) => {
            const isActive = activeBlocks.includes(i)
            return (
              <div
                key={i}
                className={`w-full aspect-square rounded-md flex items-center justify-center ${
                  isActive ? "bg-green-600 scale-110" : "bg-gray-700"
                } transition-all duration-300`}
              >
                {isActive && <Zap className="text-white w-6 h-6" />}
                <span className="text-xs text-white absolute bottom-1">Thread {i}</span>
              </div>
            )
          })}
        </div>

        <p className="text-center text-gray-300">
          CUDA organizes threads into blocks that execute in parallel across GPU cores, enabling massive parallelism for
          compute-intensive tasks.
        </p>
        <p className="text-center text-gray-400 mt-2 text-sm">{activeBlocks.length} threads active</p>
      </div>
    </div>
  )
}
