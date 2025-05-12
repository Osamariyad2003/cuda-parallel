"use client"

import { useState } from "react"

type MemoryType = "global" | "shared" | "local" | "constant" | "texture" | "registers"

export default function MemoryHierarchyDiagram() {
  const [activeMemory, setActiveMemory] = useState<MemoryType | null>(null)

  const memoryTypes: Record<
    MemoryType,
    { title: string; description: string; color: string; access: string; scope: string; size: string }
  > = {
    registers: {
      title: "Registers",
      description: "Fastest on-chip memory, private to each thread",
      color: "bg-purple-500",
      access: "Fastest (~1 clock cycle)",
      scope: "Thread",
      size: "Limited per thread (typically 255 per thread)",
    },
    shared: {
      title: "Shared Memory",
      description: "Fast on-chip memory shared by all threads in a block",
      color: "bg-blue-500",
      access: "Very Fast (~10-20 clock cycles)",
      scope: "Block",
      size: "Up to 48KB per block (varies by GPU)",
    },
    local: {
      title: "Local Memory",
      description: "Private memory for each thread, but resides in device memory",
      color: "bg-yellow-500",
      access: "Slow (hundreds of clock cycles)",
      scope: "Thread",
      size: "Limited by device memory",
    },
    constant: {
      title: "Constant Memory",
      description: "Read-only memory cached on-chip",
      color: "bg-red-500",
      access: "Fast for broadcast access patterns",
      scope: "Grid",
      size: "64KB total",
    },
    texture: {
      title: "Texture Memory",
      description: "Cached memory optimized for 2D spatial locality",
      color: "bg-orange-500",
      access: "Medium (optimized for 2D access patterns)",
      scope: "Grid",
      size: "Varies by GPU",
    },
    global: {
      title: "Global Memory",
      description: "Main device memory accessible by all threads",
      color: "bg-green-500",
      access: "Slow (hundreds of clock cycles)",
      scope: "Grid",
      size: "Several GB (varies by GPU)",
    },
  }

  return (
    <div className="w-full">
      <div className="relative w-full h-[500px] bg-gray-900 rounded-lg border border-gray-700 p-4 mb-6">
        {/* GPU Outline */}
        <div className="absolute inset-4 border-2 border-dashed border-gray-600 rounded-lg flex flex-col">
          {/* GPU Label */}
          <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-gray-900 px-4">
            <span className="text-gray-400">GPU</span>
          </div>

          {/* SM Blocks */}
          <div className="flex-1 p-4 flex flex-wrap gap-4 justify-center content-start">
            {[1, 2, 3, 4].map((sm) => (
              <div
                key={sm}
                className="w-[calc(50%-1rem)] h-[calc(50%-1rem)] border border-gray-700 rounded bg-gray-800 p-3 flex flex-col"
              >
                <div className="text-xs text-gray-400 mb-2">Streaming Multiprocessor {sm}</div>

                {/* Registers */}
                <div
                  className={`mb-2 p-2 rounded ${
                    activeMemory === "registers" ? "ring-2 ring-purple-400" : "border border-gray-700"
                  } bg-gray-900 cursor-pointer transition-all`}
                  onClick={() => setActiveMemory("registers")}
                >
                  <div className="flex items-center">
                    <div className="w-3 h-3 rounded-full bg-purple-500 mr-2"></div>
                    <span className="text-xs">Registers</span>
                  </div>
                </div>

                {/* Shared Memory */}
                <div
                  className={`p-2 rounded ${
                    activeMemory === "shared" ? "ring-2 ring-blue-400" : "border border-gray-700"
                  } bg-gray-900 cursor-pointer transition-all`}
                  onClick={() => setActiveMemory("shared")}
                >
                  <div className="flex items-center">
                    <div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div>
                    <span className="text-xs">Shared Memory</span>
                  </div>
                </div>

                {/* Local Memory (shown outside SM but referenced) */}
                <div className="mt-auto">
                  <div className="text-xs text-gray-400 flex items-center">
                    <div className="w-2 h-2 rounded-full bg-yellow-500 mr-1"></div>
                    <span>Local Memory (off-chip)</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Global Memory */}
          <div
            className={`mx-4 mb-4 p-2 rounded ${
              activeMemory === "global" ? "ring-2 ring-green-400" : "border border-gray-700"
            } bg-gray-900 cursor-pointer transition-all`}
            onClick={() => setActiveMemory("global")}
          >
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
              <span className="text-sm">Global Memory (DRAM)</span>
            </div>
          </div>

          {/* Constant and Texture Memory */}
          <div className="mx-4 mb-4 flex gap-4">
            <div
              className={`flex-1 p-2 rounded ${
                activeMemory === "constant" ? "ring-2 ring-red-400" : "border border-gray-700"
              } bg-gray-900 cursor-pointer transition-all`}
              onClick={() => setActiveMemory("constant")}
            >
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
                <span className="text-sm">Constant Memory</span>
              </div>
            </div>
            <div
              className={`flex-1 p-2 rounded ${
                activeMemory === "texture" ? "ring-2 ring-orange-400" : "border border-gray-700"
              } bg-gray-900 cursor-pointer transition-all`}
              onClick={() => setActiveMemory("texture")}
            >
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-orange-500 mr-2"></div>
                <span className="text-sm">Texture Memory</span>
              </div>
            </div>
          </div>
        </div>

        {/* Local Memory (outside GPU) */}
        <div
          className={`absolute bottom-4 left-4 w-[calc(100%-2rem)] p-2 rounded ${
            activeMemory === "local" ? "ring-2 ring-yellow-400" : "border border-gray-700"
          } bg-gray-900 cursor-pointer transition-all`}
          onClick={() => setActiveMemory("local")}
        >
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
            <span className="text-sm">Local Memory (per Thread)</span>
          </div>
        </div>

        {/* Memory Info Overlay */}
        {activeMemory && (
          <div className="absolute top-4 right-4 w-64 bg-gray-800 rounded-lg border border-gray-700 p-3 shadow-lg">
            <div className="flex items-center mb-2">
              <div className={`w-4 h-4 rounded-full ${memoryTypes[activeMemory].color} mr-2`}></div>
              <h4 className="font-medium text-white">{memoryTypes[activeMemory].title}</h4>
            </div>
            <p className="text-sm text-gray-300 mb-2">{memoryTypes[activeMemory].description}</p>
            <div className="text-xs text-gray-400 space-y-1">
              <div className="flex justify-between">
                <span>Access Speed:</span>
                <span>{memoryTypes[activeMemory].access}</span>
              </div>
              <div className="flex justify-between">
                <span>Scope:</span>
                <span>{memoryTypes[activeMemory].scope}</span>
              </div>
              <div className="flex justify-between">
                <span>Size:</span>
                <span>{memoryTypes[activeMemory].size}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="text-center text-gray-400 text-sm">
        Click on different memory types to learn more about CUDA's memory hierarchy
      </div>
    </div>
  )
}
