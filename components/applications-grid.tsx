import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Activity, Cpu, Database, Film, BarChartIcon as ChartBar } from "lucide-react"

export default function ApplicationsGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-green-400" />
            <span className="text-green-400">AI and Deep Learning</span>
          </CardTitle>
          <CardDescription>Powering the AI revolution</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-300 mb-4">
            CUDA is the foundation for major deep learning frameworks like TensorFlow, PyTorch, and MXNet, enabling
            breakthroughs in:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-gray-300">
            <li>Computer vision and image recognition</li>
            <li>Natural language processing</li>
            <li>Recommendation systems</li>
            <li>Generative AI and diffusion models</li>
            <li>Reinforcement learning</li>
          </ul>
        </CardContent>
      </Card>

      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-green-400" />
            <span className="text-green-400">Scientific Computing</span>
          </CardTitle>
          <CardDescription>Accelerating research and discovery</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-300 mb-4">CUDA accelerates complex scientific simulations and computations in:</p>
          <ul className="list-disc pl-5 space-y-1 text-gray-300">
            <li>Molecular dynamics and quantum chemistry</li>
            <li>Computational fluid dynamics</li>
            <li>Weather and climate modeling</li>
            <li>Astrophysics simulations</li>
            <li>Genomics and bioinformatics</li>
          </ul>
        </CardContent>
      </Card>

      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Film className="h-5 w-5 text-green-400" />
            <span className="text-green-400">Computer Graphics</span>
          </CardTitle>
          <CardDescription>Beyond traditional rendering</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-300 mb-4">From film production to real-time visualization, CUDA powers:</p>
          <ul className="list-disc pl-5 space-y-1 text-gray-300">
            <li>Ray tracing and path tracing</li>
            <li>Physics-based rendering</li>
            <li>Computational photography</li>
            <li>Video processing and encoding</li>
            <li>Virtual and augmented reality</li>
          </ul>
        </CardContent>
      </Card>

      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5 text-green-400" />
            <span className="text-green-400">Data Analytics</span>
          </CardTitle>
          <CardDescription>Processing massive datasets</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-300 mb-4">CUDA accelerates data processing and analytics workloads:</p>
          <ul className="list-disc pl-5 space-y-1 text-gray-300">
            <li>Database operations and SQL queries</li>
            <li>Graph analytics and network analysis</li>
            <li>Time series analysis</li>
            <li>Data mining and pattern recognition</li>
            <li>Real-time streaming analytics</li>
          </ul>
        </CardContent>
      </Card>

      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-green-400" />
            <span className="text-green-400">High-Performance Computing</span>
          </CardTitle>
          <CardDescription>Supercomputing at scale</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-300 mb-4">CUDA powers some of the world's fastest supercomputers:</p>
          <ul className="list-disc pl-5 space-y-1 text-gray-300">
            <li>Multi-GPU and multi-node computing</li>
            <li>Exascale computing initiatives</li>
            <li>Energy-efficient computing</li>
            <li>Hybrid CPU-GPU architectures</li>
            <li>Scientific visualization of massive datasets</li>
          </ul>
        </CardContent>
      </Card>

      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ChartBar className="h-5 w-5 text-green-400" />
            <span className="text-green-400">Signal Processing</span>
          </CardTitle>
          <CardDescription>Real-time analysis and transformation</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-300 mb-4">CUDA accelerates signal processing applications:</p>
          <ul className="list-disc pl-5 space-y-1 text-gray-300">
            <li>Fast Fourier Transforms (FFTs)</li>
            <li>Audio and speech processing</li>
            <li>Radar and sonar signal analysis</li>
            <li>Communications systems</li>
            <li>Medical signal processing (EEG, ECG)</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
