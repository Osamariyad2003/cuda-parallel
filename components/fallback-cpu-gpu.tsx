import { Cpu, Layers } from "lucide-react"

export default function FallbackCpuGpu() {
  return (
    <div className="w-full h-[400px] rounded-lg overflow-hidden border border-gray-700 bg-gray-900 flex flex-col items-center justify-center p-4">
      <div className="flex w-full justify-around mb-8">
        <div className="flex flex-col items-center">
          <div className="w-32 h-32 bg-gray-800 rounded-lg flex items-center justify-center mb-4">
            <Cpu className="w-16 h-16 text-gray-400" />
          </div>
          <h3 className="text-white font-medium">CPU</h3>
          <p className="text-gray-400 text-sm">Few Powerful Cores</p>
          <div className="mt-4 flex gap-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="w-8 h-8 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>

        <div className="flex flex-col items-center">
          <div className="w-32 h-32 bg-gray-800 rounded-lg flex items-center justify-center mb-4">
            <Layers className="w-16 h-16 text-green-500" />
          </div>
          <h3 className="text-white font-medium">GPU</h3>
          <p className="text-gray-400 text-sm">Thousands of Cores</p>
          <div className="mt-4 grid grid-cols-8 gap-1">
            {Array.from({ length: 32 }).map((_, i) => (
              <div key={i} className="w-2 h-2 bg-green-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>

      <p className="text-center text-gray-300 max-w-md">
        CPUs have a few powerful cores optimized for sequential processing, while GPUs have thousands of smaller cores
        designed for parallel workloads.
      </p>
    </div>
  )
}
