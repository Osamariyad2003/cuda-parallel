"use client"

import { useState } from "react"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

type Application = "matrix" | "image" | "ml" | "physics" | "finance"

interface PerformanceData {
  name: string
  cpuTime: number
  gpuTime: number
  speedup: number
  description: string
}

export default function PerformanceComparisonChart() {
  const [selectedApp, setSelectedApp] = useState<Application>("matrix")

  const applications: Record<Application, PerformanceData> = {
    matrix: {
      name: "Matrix Operations",
      cpuTime: 1000,
      gpuTime: 15,
      speedup: 66.7,
      description:
        "Matrix multiplication (1024x1024) shows dramatic speedup on GPU due to the highly parallel nature of the computation.",
    },
    image: {
      name: "Image Processing",
      cpuTime: 500,
      gpuTime: 20,
      speedup: 25,
      description:
        "Image processing operations like convolution filters benefit from GPU's ability to process pixels in parallel.",
    },
    ml: {
      name: "Deep Learning",
      cpuTime: 2000,
      gpuTime: 12,
      speedup: 166.7,
      description:
        "Neural network training with large batches shows massive speedups due to GPU's ability to handle matrix operations efficiently.",
    },
    physics: {
      name: "Physics Simulation",
      cpuTime: 800,
      gpuTime: 30,
      speedup: 26.7,
      description:
        "N-body simulations and fluid dynamics benefit from GPU's ability to compute particle interactions in parallel.",
    },
    finance: {
      name: "Financial Modeling",
      cpuTime: 600,
      gpuTime: 25,
      speedup: 24,
      description:
        "Monte Carlo simulations for options pricing show significant speedup due to the independent nature of each simulation.",
    },
  }

  const currentApp = applications[selectedApp]
  const maxTime = Math.max(...Object.values(applications).map((app) => app.cpuTime))

  return (
    <div className="w-full">
      <Tabs defaultValue="matrix" onValueChange={(value) => setSelectedApp(value as Application)}>
        <TabsList className="grid w-full grid-cols-5 bg-gray-800">
          <TabsTrigger value="matrix">Matrix</TabsTrigger>
          <TabsTrigger value="image">Image</TabsTrigger>
          <TabsTrigger value="ml">Deep Learning</TabsTrigger>
          <TabsTrigger value="physics">Physics</TabsTrigger>
          <TabsTrigger value="finance">Finance</TabsTrigger>
        </TabsList>

        <div className="mt-6 bg-gray-800 rounded-lg border border-gray-700 p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-bold text-white">{currentApp.name}</h3>
            <div className="text-2xl font-bold text-green-400">{currentApp.speedup.toFixed(1)}x Faster</div>
          </div>

          <div className="space-y-6">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-gray-400">CPU</span>
                <span className="text-sm text-gray-400">{currentApp.cpuTime} ms</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-6">
                <div
                  className="bg-gray-500 h-6 rounded-full"
                  style={{ width: `${(currentApp.cpuTime / maxTime) * 100}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-gray-400">GPU (CUDA)</span>
                <span className="text-sm text-gray-400">{currentApp.gpuTime} ms</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-6">
                <div
                  className="bg-green-500 h-6 rounded-full"
                  style={{ width: `${(currentApp.gpuTime / maxTime) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          <p className="mt-6 text-gray-300 text-sm">{currentApp.description}</p>

          <div className="mt-6 grid grid-cols-3 gap-4 text-center">
            <div className="bg-gray-900 rounded p-3">
              <div className="text-2xl font-bold text-green-400">{currentApp.speedup.toFixed(1)}x</div>
              <div className="text-xs text-gray-400">Overall Speedup</div>
            </div>
            <div className="bg-gray-900 rounded p-3">
              <div className="text-2xl font-bold text-green-400">
                {((1 - currentApp.gpuTime / currentApp.cpuTime) * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-400">Time Reduction</div>
            </div>
            <div className="bg-gray-900 rounded p-3">
              <div className="text-2xl font-bold text-green-400">
                {(currentApp.cpuTime / currentApp.gpuTime).toFixed(0)}x
              </div>
              <div className="text-xs text-gray-400">Throughput Increase</div>
            </div>
          </div>
        </div>
      </Tabs>
    </div>
  )
}
