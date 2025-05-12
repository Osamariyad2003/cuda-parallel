"use client"

import { useState, useEffect } from "react"
import { Zap, Play, Pause, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"

export default function EnhancedFallbackParallelism() {
  const [activeBlocks, setActiveBlocks] = useState<number[]>([])
  const [blockData, setBlockData] = useState<number[]>(Array(16).fill(0))
  const [isPaused, setIsPaused] = useState(false)
  const [speed, setSpeed] = useState(800) // milliseconds between updates
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null)

  useEffect(() => {
    if (isPaused) return

    // Simulate parallel processing by activating blocks in waves
    const interval = setInterval(() => {
      const newActiveBlocks: number[] = []
      const count = Math.floor(Math.random() * 8) + 4 // Activate 4-12 blocks at random

      // Generate new random data for each block
      const newBlockData = [...blockData]

      for (let i = 0; i < count; i++) {
        const blockIndex = Math.floor(Math.random() * 16)
        if (!newActiveBlocks.includes(blockIndex)) {
          newActiveBlocks.push(blockIndex)
          newBlockData[blockIndex] = Math.floor(Math.random() * 100)
        }
      }

      setActiveBlocks(newActiveBlocks)
      setBlockData(newBlockData)
    }, speed)

    return () => clearInterval(interval)
  }, [isPaused, speed, blockData])

  const togglePause = () => {
    setIsPaused(!isPaused)
  }

  const resetSimulation = () => {
    setActiveBlocks([])
    setBlockData(Array(16).fill(0))
    setIsPaused(false)
  }

  const handleBlockClick = (index: number) => {
    setSelectedBlock(selectedBlock === index ? null : index)
  }

  return (
    <div className="w-full h-[400px] rounded-lg overflow-hidden border border-gray-700 bg-gray-900 flex flex-col p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-medium">CUDA Thread Blocks</h3>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0 border-gray-700"
            onClick={togglePause}
            title={isPaused ? "Resume" : "Pause"}
          >
            {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 w-8 p-0 border-gray-700"
            onClick={resetSimulation}
            title="Reset"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-4 flex-1">
        {Array.from({ length: 16 }).map((_, i) => {
          const isActive = activeBlocks.includes(i)
          const isSelected = selectedBlock === i
          return (
            <div
              key={i}
              className={`relative rounded-md flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                isActive ? "bg-green-600" : "bg-gray-700"
              } ${isSelected ? "ring-2 ring-white" : ""} ${isActive ? "scale-105" : ""}`}
              onClick={() => handleBlockClick(i)}
            >
              {isActive && <Zap className="text-white w-6 h-6 mb-2" />}
              <span className="text-xs text-white">Thread {i}</span>
              {isActive && <span className="text-xs text-white mt-1">Data: {blockData[i]}</span>}
            </div>
          )
        })}
      </div>

      {selectedBlock !== null && (
        <div className="bg-gray-800 p-3 rounded-md mb-4">
          <h4 className="text-sm font-medium text-white mb-1">Thread {selectedBlock} Details</h4>
          <div className="grid grid-cols-2 gap-2 text-xs text-gray-300">
            <div>Status: {activeBlocks.includes(selectedBlock) ? "Active" : "Idle"}</div>
            <div>Block ID: {Math.floor(selectedBlock / 4)}</div>
            <div>Thread ID: {selectedBlock % 4}</div>
            <div>Data: {activeBlocks.includes(selectedBlock) ? blockData[selectedBlock] : "N/A"}</div>
          </div>
        </div>
      )}

      <div className="flex items-center gap-4">
        <span className="text-xs text-gray-400">Speed:</span>
        <Slider
          value={[speed]}
          min={100}
          max={2000}
          step={100}
          onValueChange={(value) => setSpeed(value[0])}
          className="flex-1"
        />
        <span className="text-xs text-gray-400 w-12">{speed}ms</span>
      </div>

      <div className="mt-4 flex justify-between text-xs text-gray-400">
        <div>{activeBlocks.length} threads active</div>
        <div>{((activeBlocks.length / 16) * 100).toFixed(1)}% utilization</div>
      </div>
    </div>
  )
}
