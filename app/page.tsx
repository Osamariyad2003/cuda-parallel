"use client"

import Link from "next/link"
import { ArrowRight, Cpu, Layers, Zap, Download, Code, BookOpen, Rocket, ExternalLink } from "lucide-react"

import { Button } from "@/components/ui/button"
import FallbackCudaHero from "@/components/fallback-cuda-hero"
import EnhancedFallbackParallelism from "@/components/enhanced-fallback-parallelism"
import FallbackCpuGpu from "@/components/fallback-cpu-gpu"
import ApplicationsGrid from "@/components/applications-grid"
import MemoryHierarchyDiagram from "@/components/memory-hierarchy-diagram"
import PerformanceComparisonChart from "@/components/performance-comparison-chart"
import CudaTutorial from "@/components/cuda-tutorial"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-black text-white">
      {/* Navigation */}
      <header className="sticky top-0 z-50 w-full border-b border-gray-800 bg-black/80 backdrop-blur-sm">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-green-500">
            <Cpu className="h-5 w-5" />
            <span>CUDA Parallelism</span>
          </div>
          <nav className="hidden md:flex gap-6">
            <Link href="#overview" className="text-sm font-medium hover:text-green-400 transition-colors">
              Overview
            </Link>
            <Link href="#parallelism" className="text-sm font-medium hover:text-green-400 transition-colors">
              Parallelism
            </Link>
            <Link href="#architecture" className="text-sm font-medium hover:text-green-400 transition-colors">
              Architecture
            </Link>
            <Link href="#applications" className="text-sm font-medium hover:text-green-400 transition-colors">
              Applications
            </Link>
            <Link href="#performance" className="text-sm font-medium hover:text-green-400 transition-colors">
              Performance
            </Link>
            <Link href="#tutorial" className="text-sm font-medium hover:text-green-400 transition-colors">
              Tutorial
            </Link>
          </nav>
          <Button
            variant="outline"
            className="border-green-500 text-green-500 hover:bg-green-500/10"
            onClick={() => window.open("https://docs.nvidia.com/cuda/", "_blank")}
          >
            Resources <ExternalLink className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Hero Section with Visualization */}
      <section className="relative h-[80vh] overflow-hidden">
        <FallbackCudaHero />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="container px-4 md:px-6 text-center">
            <h1 className="text-4xl md:text-6xl font-bold tracking-tighter mb-4 bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-green-600">
              Understanding CUDA Parallelism
            </h1>
            <p className="max-w-[700px] mx-auto text-gray-400 md:text-xl mb-8">
              Explore how NVIDIA's CUDA architecture enables massive parallel computing power for scientific and AI
              applications
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                className="bg-green-500 hover:bg-green-600 text-black"
                onClick={() => document.getElementById("overview")?.scrollIntoView({ behavior: "smooth" })}
              >
                Start Learning <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                className="border-green-500 text-green-500 hover:bg-green-500/10"
                onClick={() => window.open("https://developer.nvidia.com/cuda-downloads", "_blank")}
              >
                Download CUDA Toolkit <Download className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Overview Section */}
      <section id="overview" className="py-16 bg-gradient-to-b from-black to-gray-900">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter mb-8 text-center text-green-400">What is CUDA?</h2>
          <div className="max-w-3xl mx-auto text-gray-300 mb-12 text-center">
            <p className="mb-4">
              CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on
              graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing
              applications by harnessing the power of GPUs.
            </p>
            <p>
              In GPU-accelerated applications, the sequential part of the workload runs on the CPU – which is optimized
              for single-threaded performance – while the compute intensive portion of the application runs on thousands
              of GPU cores in parallel.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-green-400">Compute Unified Device Architecture</CardTitle>
                <CardDescription>NVIDIA's parallel computing platform</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300">
                  CUDA enables developers to program in popular languages such as C, C++, Fortran, Python and MATLAB and
                  express parallelism through extensions in the form of a few basic keywords.
                </p>
              </CardContent>
            </Card>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-green-400">CUDA Toolkit</CardTitle>
                <CardDescription>Everything you need for GPU development</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300">
                  The CUDA Toolkit from NVIDIA provides everything you need to develop GPU-accelerated applications. The
                  toolkit includes GPU-accelerated libraries, a compiler, development tools and the CUDA runtime.
                </p>
              </CardContent>
            </Card>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-green-400">Widespread Adoption</CardTitle>
                <CardDescription>Used across industries</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300">
                  Thousands of applications developed with CUDA have been deployed to GPUs in embedded systems,
                  workstations, datacenters and in the cloud, powering innovations across industries.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CPU vs GPU Section */}
      <section className="py-16 bg-gray-900">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col md:flex-row gap-12 items-center">
            <div className="md:w-1/2">
              <h2 className="text-3xl font-bold tracking-tighter mb-4 text-green-400">CPU vs GPU Architecture</h2>
              <p className="text-gray-300 mb-6">
                CPUs and GPUs are designed for different computing tasks. CPUs excel at handling sequential tasks with
                complex logic, while GPUs are optimized for parallel processing of simpler tasks across thousands of
                cores.
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="bg-green-500 p-2 rounded-full">
                    <Cpu className="h-5 w-5 text-black" />
                  </div>
                  <div>
                    <h3 className="font-medium text-white">CPU: Few Powerful Cores</h3>
                    <p className="text-gray-400">
                      Optimized for single-threaded performance with complex caching and control flow
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-green-500 p-2 rounded-full">
                    <Layers className="h-5 w-5 text-black" />
                  </div>
                  <div>
                    <h3 className="font-medium text-white">GPU: Thousands of Cores</h3>
                    <p className="text-gray-400">
                      Designed for massive parallelism with simpler cores but much higher throughput
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="md:w-1/2 h-[400px]">
              <FallbackCpuGpu />
            </div>
          </div>
        </div>
      </section>

      {/* Parallelism Demo Section */}
      <section id="parallelism" className="py-16 bg-gray-900">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col md:flex-row gap-12 items-center">
            <div className="md:w-1/2 h-[400px] order-2 md:order-1">
              <EnhancedFallbackParallelism />
            </div>
            <div className="md:w-1/2 order-1 md:order-2">
              <h2 className="text-3xl font-bold tracking-tighter mb-4 text-green-400">Visualizing CUDA Parallelism</h2>
              <p className="text-gray-300 mb-6">
                CUDA's architecture organizes threads into blocks, and blocks into grids. This hierarchical structure
                enables efficient parallel execution across thousands of cores.
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="bg-green-500 p-2 rounded-full">
                    <Layers className="h-5 w-5 text-black" />
                  </div>
                  <div>
                    <h3 className="font-medium text-white">Thread Hierarchy</h3>
                    <p className="text-gray-400">Threads, blocks, and grids form the execution model</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-green-500 p-2 rounded-full">
                    <Zap className="h-5 w-5 text-black" />
                  </div>
                  <div>
                    <h3 className="font-medium text-white">SIMT Architecture</h3>
                    <p className="text-gray-400">Single Instruction, Multiple Thread execution</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Section */}
      <section id="architecture" className="py-16 bg-gradient-to-b from-gray-900 to-black">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter mb-12 text-center text-green-400">CUDA Architecture</h2>

          <div className="mb-12">
            <h3 className="text-xl font-bold mb-6 text-white text-center">Memory Hierarchy</h3>
            <MemoryHierarchyDiagram />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-xl font-bold mb-4 text-white">Thread Hierarchy</h3>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-green-500"></span>
                  <span>Thread: Individual execution unit that runs a kernel</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                  <span>Block: Group of threads that can cooperate via shared memory</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                  <span>Grid: Collection of blocks that execute the same kernel</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
                  <span>Warp: Group of 32 threads executed together (SIMT)</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-red-500"></span>
                  <span>Streaming Multiprocessor (SM): Hardware unit that executes warps</span>
                </li>
              </ul>
            </div>
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-xl font-bold mb-4 text-white">Execution Model</h3>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-green-500"></span>
                  <span>Kernels: Functions executed on the GPU</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                  <span>SIMT: Single Instruction, Multiple Thread execution model</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                  <span>Synchronization: Threads in a block can synchronize execution</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
                  <span>Occupancy: Ratio of active warps to maximum possible</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-red-500"></span>
                  <span>Grid/Block Dimensions: Define thread organization</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Applications Section */}
      <section id="applications" className="py-16 bg-black">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter mb-8 text-center text-green-400">
            Applications Developed with CUDA
          </h2>
          <p className="text-center text-gray-300 max-w-3xl mx-auto mb-12">
            Thousands of applications developed with CUDA have been deployed to GPUs in embedded systems, workstations,
            datacenters and in the cloud.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
            <div className="bg-gray-800 p-4 rounded-lg flex flex-col items-center justify-center">
              <img
                src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Adobe_Systems_logo_and_wordmark.svg/1200px-Adobe_Systems_logo_and_wordmark.svg.png"
                alt="Adobe logo"
                className="h-12 object-contain mb-2"
              />
              <span className="text-sm text-gray-300">Adobe</span>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg flex flex-col items-center justify-center">
              <img
                src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Ansys_logo_%282019%29.svg/1200px-Ansys_logo_%282019%29.svg.png"
                alt="Ansys logo"
                className="h-12 object-contain mb-2"
              />
              <span className="text-sm text-gray-300">Ansys</span>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg flex flex-col items-center justify-center">
              <img
                src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Autodesk_Logo.svg/1200px-Autodesk_Logo.svg.png"
                alt="Autodesk logo"
                className="h-12 object-contain mb-2"
              />
              <span className="text-sm text-gray-300">Autodesk</span>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg flex flex-col items-center justify-center">
              <img
                src="https://upload.wikimedia.org/wikipedia/en/thumb/9/9a/Dassault_Syst%C3%A8mes_logo.svg/1200px-Dassault_Syst%C3%A8mes_logo.svg.png"
                alt="Dassault Systèmes logo"
                className="h-12 object-contain mb-2"
              />
              <span className="text-sm text-gray-300">Dassault Systèmes</span>
            </div>
          </div>

          <ApplicationsGrid />

          <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-green-400">Healthcare & Medical Research</CardTitle>
                <CardDescription>Accelerating medical breakthroughs</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300 mb-4">
                  CUDA accelerates medical imaging, drug discovery, genomics, and patient data analysis. Medical
                  researchers use CUDA to:
                </p>
                <ul className="list-disc pl-5 space-y-1 text-gray-300">
                  <li>Process MRI and CT scans up to 10x faster</li>
                  <li>Simulate protein folding for drug development</li>
                  <li>Analyze genomic data for personalized medicine</li>
                  <li>Train AI models on medical images for disease detection</li>
                </ul>
              </CardContent>
              <CardFooter>
                <Button
                  variant="outline"
                  className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                  onClick={() => window.open("https://developer.nvidia.com/healthcare-solutions", "_blank")}
                >
                  Learn More <ExternalLink className="ml-2 h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-green-400">Financial Services</CardTitle>
                <CardDescription>Powering high-frequency trading and risk analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300 mb-4">
                  Financial institutions leverage CUDA for complex calculations and real-time analytics:
                </p>
                <ul className="list-disc pl-5 space-y-1 text-gray-300">
                  <li>Risk assessment and portfolio optimization</li>
                  <li>High-frequency trading algorithms</li>
                  <li>Options pricing and derivatives valuation</li>
                  <li>Fraud detection using machine learning</li>
                </ul>
              </CardContent>
              <CardFooter>
                <Button
                  variant="outline"
                  className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                  onClick={() => window.open("https://developer.nvidia.com/finance-solutions", "_blank")}
                >
                  Learn More <ExternalLink className="ml-2 h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          </div>
        </div>
      </section>

      {/* Performance Comparison Section */}
      <section id="performance" className="py-16 bg-gray-900">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter mb-8 text-center text-green-400">
            CUDA Performance Gains
          </h2>
          <p className="text-center text-gray-300 max-w-3xl mx-auto mb-12">
            CUDA-accelerated applications show dramatic performance improvements across various domains.
          </p>

          <PerformanceComparisonChart />
        </div>
      </section>

      {/* Tutorial Section */}
      <section id="tutorial" className="py-16 bg-black">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter mb-8 text-center text-green-400">
            CUDA Programming Tutorial
          </h2>
          <p className="text-center text-gray-300 max-w-3xl mx-auto mb-12">
            Learn how to write CUDA code with this step-by-step tutorial. Follow along to understand the basics of GPU
            programming.
          </p>

          <CudaTutorial />
        </div>
      </section>

      {/* Examples Section */}
      <section id="examples" className="py-16 bg-gradient-to-b from-black to-gray-900">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter mb-8 text-center text-green-400">CUDA Code Examples</h2>

          <Tabs defaultValue="vector" className="w-full max-w-4xl mx-auto">
            <TabsList className="grid w-full grid-cols-3 bg-gray-800">
              <TabsTrigger value="vector">Vector Addition</TabsTrigger>
              <TabsTrigger value="matrix">Matrix Multiplication</TabsTrigger>
              <TabsTrigger value="image">Image Processing</TabsTrigger>
            </TabsList>
            <TabsContent value="vector" className="mt-4">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-green-400">Vector Addition</CardTitle>
                  <CardDescription>Basic parallel vector addition in CUDA</CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-gray-900 p-4 rounded-md overflow-x-auto text-sm text-gray-300">
                    {`// CUDA Kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  // Get global thread ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Make sure we don't go out of bounds
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  // Vector size
  int n = 1000000;
  size_t bytes = n * sizeof(float);
  
  // Host vectors
  float *h_a, *h_b, *h_c;
  
  // Device vectors
  float *d_a, *d_b, *d_c;
  
  // Allocate host memory
  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);
  h_c = (float*)malloc(bytes);
  
  // Initialize vectors
  for (int i = 0; i < n; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }
  
  // Allocate device memory
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  
  // Copy data from host to device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  
  // Set grid and block dimensions
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  
  // Launch kernel
  vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
  
  // Copy result back to host
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  // Free host memory
  free(h_a);
  free(h_b);
  free(h_c);
  
  return 0;
}`}
                  </pre>
                </CardContent>
                <CardFooter>
                  <Button
                    variant="outline"
                    className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                    onClick={() =>
                      window.open("https://developer.nvidia.com/blog/even-easier-introduction-cuda/", "_blank")
                    }
                  >
                    Learn More About CUDA Kernels <ExternalLink className="ml-2 h-4 w-4" />
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>
            <TabsContent value="matrix" className="mt-4">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-green-400">Matrix Multiplication</CardTitle>
                  <CardDescription>Parallel matrix multiplication in CUDA</CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-gray-900 p-4 rounded-md overflow-x-auto text-sm text-gray-300">
                    {`// CUDA Kernel for matrix multiplication
__global__ void matrixMul(float* A, float* B, float* C, int N) {
  // Calculate row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Check if within matrix bounds
  if (row < N && col < N) {
    float sum = 0.0f;
    
    // Compute dot product of row of A and column of B
    for (int i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * N + col];
    }
    
    // Store result in C
    C[row * N + col] = sum;
  }
}

int main() {
  int N = 1024; // Matrix dimensions
  size_t bytes = N * N * sizeof(float);
  
  // Host matrices
  float *h_A, *h_B, *h_C;
  
  // Device matrices
  float *d_A, *d_B, *d_C;
  
  // Allocate host memory
  h_A = (float*)malloc(bytes);
  h_B = (float*)malloc(bytes);
  h_C = (float*)malloc(bytes);
  
  // Initialize matrices
  for (int i = 0; i < N * N; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }
  
  // Allocate device memory
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);
  
  // Copy data from host to device
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  
  // Set grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
  
  // Launch kernel
  matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  
  // Copy result back to host
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  
  return 0;
}`}
                  </pre>
                </CardContent>
                <CardFooter>
                  <Button
                    variant="outline"
                    className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                    onClick={() =>
                      window.open(
                        "https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/",
                        "_blank",
                      )
                    }
                  >
                    Explore Matrix Operations in CUDA <ExternalLink className="ml-2 h-4 w-4" />
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>
            <TabsContent value="image" className="mt-4">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-green-400">Image Processing</CardTitle>
                  <CardDescription>Grayscale conversion in CUDA</CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-gray-900 p-4 rounded-md overflow-x-auto text-sm text-gray-300">
                    {`// CUDA Kernel for grayscale conversion
__global__ void rgbToGrayscale(uchar4* rgbImage, unsigned char* grayImage, int width, int height) {
  // Calculate pixel position
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Check if within image bounds
  if (x < width && y < height) {
    // Get pixel index
    int idx = y * width + x;
    
    // Get RGB values
    uchar4 pixel = rgbImage[idx];
    unsigned char r = pixel.x;
    unsigned char g = pixel.y;
    unsigned char b = pixel.z;
    
    // Convert to grayscale using luminance method
    // Y = 0.299*R + 0.587*G + 0.114*B
    grayImage[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
  }
}

int main() {
  // Image dimensions
  int width = 1920;
  int height = 1080;
  
  // Allocate host memory
  uchar4* h_rgbImage = (uchar4*)malloc(width * height * sizeof(uchar4));
  unsigned char* h_grayImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));
  
  // Initialize image (in a real application, load from file)
  for (int i = 0; i < width * height; i++) {
    h_rgbImage[i] = make_uchar4(rand() % 256, rand() % 256, rand() % 256, 255);
  }
  
  // Allocate device memory
  uchar4* d_rgbImage;
  unsigned char* d_grayImage;
  cudaMalloc(&d_rgbImage, width * height * sizeof(uchar4));
  cudaMalloc(&d_grayImage, width * height * sizeof(unsigned char));
  
  // Copy input image to device
  cudaMemcpy(d_rgbImage, h_rgbImage, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
  
  // Set grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
  
  // Launch kernel
  rgbToGrayscale<<<gridDim, blockDim>>>(d_rgbImage, d_grayImage, width, height);
  
  // Copy result back to host
  cudaMemcpy(h_grayImage, d_grayImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_rgbImage);
  cudaFree(d_grayImage);
  
  // Free host memory
  free(h_rgbImage);
  free(h_grayImage);
  
  return 0;
}`}
                  </pre>
                </CardContent>
                <CardFooter>
                  <Button
                    variant="outline"
                    className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                    onClick={() =>
                      window.open("https://developer.nvidia.com/blog/accelerated-computing-image-processing/", "_blank")
                    }
                  >
                    Explore Image Processing with CUDA <ExternalLink className="ml-2 h-4 w-4" />
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* Resources Section */}
      <section className="py-16 bg-black">
        <div className="container px-4 md:px-6">
          <h2 className="text-3xl font-bold tracking-tighter mb-8 text-center text-green-400">Learning Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="h-5 w-5 text-green-400" />
                  <span className="text-green-400">Documentation</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300 mb-4">
                  Access comprehensive CUDA documentation, including programming guides, API references, and best
                  practices.
                </p>
                <Button
                  variant="outline"
                  className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                  onClick={() => window.open("https://docs.nvidia.com/cuda/", "_blank")}
                >
                  View Documentation <ExternalLink className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Code className="h-5 w-5 text-green-400" />
                  <span className="text-green-400">Code Samples</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300 mb-4">
                  Explore a wide range of CUDA code samples covering various domains and optimization techniques.
                </p>
                <Button
                  variant="outline"
                  className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                  onClick={() => window.open("https://github.com/NVIDIA/cuda-samples", "_blank")}
                >
                  Browse Samples <ExternalLink className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Rocket className="h-5 w-5 text-green-400" />
                  <span className="text-green-400">Tutorials</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300 mb-4">
                  Follow step-by-step tutorials to learn CUDA programming from basics to advanced techniques.
                </p>
                <Button
                  variant="outline"
                  className="w-full border-green-500 text-green-500 hover:bg-green-500/10"
                  onClick={() => window.open("https://developer.nvidia.com/cuda-education-training", "_blank")}
                >
                  Start Learning <ExternalLink className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Download Section */}
      <section className="py-16 bg-black">
        <div className="container px-4 md:px-6">
          <div className="bg-gradient-to-r from-green-900/50 to-green-700/50 rounded-lg p-8 text-center">
            <h2 className="text-3xl font-bold tracking-tighter mb-4 text-white">Ready to Get Started?</h2>
            <p className="text-gray-200 mb-8 max-w-2xl mx-auto">
              Download the CUDA Toolkit to start developing GPU-accelerated applications. The toolkit includes
              GPU-accelerated libraries, a compiler, development tools and the CUDA runtime.
            </p>
            <Button
              className="bg-green-500 hover:bg-green-600 text-black"
              onClick={() => window.open("https://developer.nvidia.com/cuda-downloads", "_blank")}
            >
              Download CUDA Toolkit <Download className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-6 border-t border-gray-800">
        <div className="container px-4 md:px-6 flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center gap-2 font-bold text-green-500 mb-4 md:mb-0">
            <Cpu className="h-5 w-5" />
            <span>CUDA Parallelism</span>
          </div>
          <div className="flex gap-6">
            <a
              href="https://developer.nvidia.com/cuda-zone"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-green-400 transition-colors"
            >
              CUDA Zone
            </a>
            <a
              href="https://docs.nvidia.com/cuda/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-green-400 transition-colors"
            >
              Documentation
            </a>
            <a
              href="https://developer.nvidia.com/cuda-education-training"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-green-400 transition-colors"
            >
              Training
            </a>
          </div>
          <p className="text-gray-400 text-sm mt-4 md:mt-0">
            © {new Date().getFullYear()} CUDA Lecture Series. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  )
}
