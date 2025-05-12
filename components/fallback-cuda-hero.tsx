import { Cpu } from "lucide-react"

export default function FallbackCudaHero() {
  return (
    <div className="absolute inset-0 bg-gradient-to-b from-black to-gray-900 flex items-center justify-center">
      <div className="absolute inset-0 opacity-20">
        <div className="grid grid-cols-8 grid-rows-8 gap-4 p-8 h-full">
          {Array.from({ length: 64 }).map((_, i) => {
            const row = Math.floor(i / 8)
            const col = i % 8
            const delay = (row + col) * 0.1

            return (
              <div
                key={i}
                className="bg-green-500 rounded-md animate-pulse"
                style={{
                  animationDelay: `${delay}s`,
                  opacity: 0.3 + (row / 8) * 0.7,
                }}
              />
            )
          })}
        </div>
      </div>

      <div className="relative z-10 flex items-center justify-center">
        <Cpu className="text-green-500 w-24 h-24 animate-pulse" />
      </div>
    </div>
  )
}
