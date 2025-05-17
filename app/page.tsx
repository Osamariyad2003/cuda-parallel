import CudaBasicsTutorial from "@/components/cuda-basics-tutorial"
import CudaSharedMemoryTutorial from "@/components/cuda-shared-memory-tutorial"
import CudaOptimizationPart1 from "@/components/cuda-optimization-part1"
import CudaOptimizationPart2 from "@/components/cuda-optimization-part2"
import CudaAtomicsTutorial from "@/components/cuda-atomics-tutorial"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowRight, Download, Layers, Cpu, Zap, Play } from "lucide-react"
import Link from "next/link"
import CudaCompilerGuide from "@/components/cuda-compiler-guide"
import CudaChatbot from "@/components/cuda-chatbot"

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Navigation Bar */}
      <nav className="border-b border-gray-800 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="flex items-center text-green-400 font-bold text-xl">
                <Zap className="mr-2 h-6 w-6" />
                CUDA Parallelism
              </Link>
              <div className="hidden md:ml-10 md:flex md:space-x-8">
                <Link href="#overview" className="text-gray-300 hover:text-white px-3 py-2">
                  Overview
                </Link>
                <Link href="#parallelism" className="text-gray-300 hover:text-white px-3 py-2">
                  Parallelism
                </Link>
                <Link href="#architecture" className="text-gray-300 hover:text-white px-3 py-2">
                  Architecture
                </Link>
                <Link href="#applications" className="text-gray-300 hover:text-white px-3 py-2">
                  Applications
                </Link>
                <Link href="#performance" className="text-gray-300 hover:text-white px-3 py-2">
                  Performance
                </Link>
                <Link href="#tutorial" className="text-white font-medium px-3 py-2 border-b-2 border-green-400">
                  Tutorial
                </Link>
                <Link href="/code-simulator-page" className="text-gray-300 hover:text-white px-3 py-2">
                  Code Simulator
                </Link>
              </div>
            </div>
            <div className="flex items-center">
              <Link href="#resources" className="text-gray-300 hover:text-white px-3 py-2 flex items-center">
                Resources
                <svg
                  className="ml-1 h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                  />
                </svg>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative bg-gray-900 overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 bg-gray-900 sm:pb-16 md:pb-20 lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 xl:mt-28">
              <div className="text-center">
                <h1 className="text-4xl tracking-tight font-extrabold text-white sm:text-5xl md:text-6xl">
                  <span className="block text-green-400">Understanding CUDA Parallelism</span>
                </h1>
                <p className="mt-3 max-w-md mx-auto text-base text-gray-300 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
                  Explore how NVIDIA's CUDA architecture enables massive parallel computing power for scientific and AI
                  applications
                </p>
                <div className="mt-5 max-w-md mx-auto sm:flex sm:justify-center md:mt-8">
                  <div className="rounded-md shadow">
                    <a
                      href="#tutorial"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-black bg-green-400 hover:bg-green-500 md:py-4 md:text-lg md:px-10"
                    >
                      Start Learning
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </a>
                  </div>
                  <div className="mt-3 rounded-md shadow sm:mt-0 sm:ml-3">
                    <a
                      href="#"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-green-400 bg-gray-800 hover:bg-gray-700 md:py-4 md:text-lg md:px-10"
                    >
                      Download CUDA Toolkit
                      <Download className="ml-2 h-5 w-5" />
                    </a>
                  </div>
                  <div className="mt-3 rounded-md shadow sm:mt-0 sm:ml-3">
                    <Link
                      href="/code-simulator-page"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-gray-700 hover:bg-gray-600 md:py-4 md:text-lg md:px-10"
                    >
                      Try Code Simulator
                      <Play className="ml-2 h-5 w-5" />
                    </Link>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>

      {/* What is CUDA Section */}
      <section id="overview" className="py-12 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-green-400 sm:text-4xl">What is CUDA?</h2>
            <p className="mt-4 max-w-3xl mx-auto text-xl text-gray-300">
              CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on
              graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing
              applications by harnessing the power of GPUs.
            </p>
            <p className="mt-4 max-w-3xl mx-auto text-lg text-gray-300">
              In GPU-accelerated applications, the sequential part of the workload runs on the CPU – which is optimized
              for single-threaded performance – while the compute intensive portion of the application runs on thousands
              of GPU cores in parallel.
            </p>
          </div>

          <div className="mt-10">
            <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
              <div className="bg-gray-800 rounded-lg shadow-lg overflow-hidden">
                <div className="px-6 py-8">
                  <h3 className="text-xl font-semibold text-green-400">Compute Unified Device Architecture</h3>
                  <p className="mt-2 text-gray-300">NVIDIA's parallel computing platform</p>
                  <p className="mt-4 text-gray-300">
                    CUDA enables developers to program in popular languages such as C, C++, Fortran, Python and MATLAB
                    and express parallelism through extensions in the form of a few basic keywords.
                  </p>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg shadow-lg overflow-hidden">
                <div className="px-6 py-8">
                  <h3 className="text-xl font-semibold text-green-400">CUDA Toolkit</h3>
                  <p className="mt-2 text-gray-300">Everything you need for GPU development</p>
                  <p className="mt-4 text-gray-300">
                    The CUDA Toolkit from NVIDIA provides everything you need to develop GPU-accelerated applications.
                    The toolkit includes GPU-accelerated libraries, a compiler, development tools and the CUDA runtime.
                  </p>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg shadow-lg overflow-hidden">
                <div className="px-6 py-8">
                  <h3 className="text-xl font-semibold text-green-400">Widespread Adoption</h3>
                  <p className="mt-2 text-gray-300">Used across industries</p>
                  <p className="mt-4 text-gray-300">
                    Thousands of applications developed with CUDA have been deployed to GPUs in embedded systems,
                    workstations, datacenters and in the cloud, powering innovations across industries.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Visualizing CUDA Parallelism */}
      <section id="parallelism" className="py-12 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-2 lg:gap-8 items-center">
            <div className="mt-10 lg:mt-0">
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-lg font-medium text-white mb-2">CUDA Thread Blocks</h3>
                <div className="grid grid-cols-4 gap-2">
                  {Array(16)
                    .fill(0)
                    .map((_, i) => (
                      <div
                        key={i}
                        className={`p-4 rounded-md flex items-center justify-center ${[3, 4, 5, 8, 10, 12, 13].includes(i) ? "bg-green-500" : "bg-gray-700"}`}
                      >
                        <div className="text-xs">
                          <div>Thread {i}</div>
                          {[3, 4, 5, 8, 10, 12, 13].includes(i) && (
                            <div>Data: {Math.floor(Math.random() * 90) + 10}</div>
                          )}
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
            <div>
              <h2 className="text-3xl font-extrabold text-green-400 sm:text-4xl">Visualizing CUDA Parallelism</h2>
              <p className="mt-3 text-lg text-gray-300">
                CUDA's architecture organizes threads into blocks, and blocks into grids. This hierarchical structure
                enables efficient parallel execution across thousands of cores.
              </p>
              <div className="mt-8 space-y-6">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <div className="flex items-center justify-center h-12 w-12 rounded-md bg-green-500 text-white">
                      <Layers className="h-6 w-6" />
                    </div>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-lg leading-6 font-medium text-white">Thread Hierarchy</h3>
                    <p className="mt-2 text-base text-gray-300">Threads, blocks, and grids form the execution model</p>
                  </div>
                </div>
                <div className="flex">
                  <div className="flex-shrink-0">
                    <div className="flex items-center justify-center h-12 w-12 rounded-md bg-green-500 text-white">
                      <Cpu className="h-6 w-6" />
                    </div>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-lg leading-6 font-medium text-white">SIMT Architecture</h3>
                    <p className="mt-2 text-base text-gray-300">Single Instruction, Multiple Thread execution</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Memory Hierarchy */}
      <section id="architecture" className="py-12 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-extrabold text-green-400 sm:text-4xl">CUDA Architecture</h2>
            <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-300">Memory Hierarchy</p>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="text-center text-gray-300 mb-4">GPU</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[1, 2, 3, 4].map((sm) => (
                <div key={sm} className="bg-gray-700 rounded-lg p-4">
                  <div className="text-sm text-gray-300 mb-2">Streaming Multiprocessor {sm}</div>
                  <div className="space-y-2">
                    <div className="bg-gray-600 rounded p-2 flex items-center">
                      <div className="h-3 w-3 rounded-full bg-purple-500 mr-2"></div>
                      <div className="text-sm text-gray-300">Registers</div>
                    </div>
                    <div className="bg-gray-600 rounded p-2 flex items-center">
                      <div className="h-3 w-3 rounded-full bg-blue-500 mr-2"></div>
                      <div className="text-sm text-gray-300">Shared Memory</div>
                    </div>
                    <div className="bg-gray-600 rounded p-2 flex items-center">
                      <div className="h-3 w-3 rounded-full bg-yellow-500 mr-2"></div>
                      <div className="text-sm text-gray-300">Local Memory (off-chip)</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 bg-gray-700 rounded-lg p-2 flex items-center">
              <div className="h-3 w-3 rounded-full bg-green-500 mr-2"></div>
              <div className="text-sm text-gray-300">Global Memory (DRAM)</div>
            </div>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-700 rounded-lg p-2 flex items-center">
                <div className="h-3 w-3 rounded-full bg-red-500 mr-2"></div>
                <div className="text-sm text-gray-300">Constant Memory</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-2 flex items-center">
                <div className="h-3 w-3 rounded-full bg-orange-500 mr-2"></div>
                <div className="text-sm text-gray-300">Texture Memory</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Tutorial Section */}
      <section id="tutorial" className="py-12 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-extrabold text-green-400 sm:text-4xl">CUDA Programming Tutorial</h2>
            <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-300">
              Learn how to write CUDA code with this step-by-step tutorial. Follow along to understand the basics of GPU
              programming.
            </p>
          </div>

          <Tabs defaultValue="basics" className="w-full">
            <TabsList className="grid w-full grid-cols-2 md:grid-cols-6 bg-gray-800">
              <TabsTrigger value="basics">CUDA Basics</TabsTrigger>
              <TabsTrigger value="shared-memory">Shared Memory</TabsTrigger>
              <TabsTrigger value="optimization1">Optimization I</TabsTrigger>
              <TabsTrigger value="optimization2">Optimization II</TabsTrigger>
              <TabsTrigger value="atomics">Atomics</TabsTrigger>
              <TabsTrigger value="compilation">Compilation</TabsTrigger>
            </TabsList>

            <TabsContent value="basics" className="mt-6">
              <h2 className="text-2xl font-bold mb-4 text-green-400">CUDA C++ Basics</h2>
              <p className="text-gray-300 mb-6">
                Learn the fundamentals of CUDA programming, including host and device code, memory management, and
                thread organization.
              </p>
              <CudaBasicsTutorial />
            </TabsContent>

            <TabsContent value="shared-memory" className="mt-6">
              <h2 className="text-2xl font-bold mb-4 text-green-400">CUDA Shared Memory</h2>
              <p className="text-gray-300 mb-6">
                Explore shared memory usage to reduce global memory accesses, implement stencil operations, and avoid
                race conditions.
              </p>
              <CudaSharedMemoryTutorial />
            </TabsContent>

            <TabsContent value="optimization1" className="mt-6">
              <h2 className="text-2xl font-bold mb-4 text-green-400">CUDA Optimization - Part 1</h2>
              <p className="text-gray-300 mb-6">
                Understand GPU architecture, memory hierarchy, execution model, and how to configure kernel launches for
                optimal performance.
              </p>
              <CudaOptimizationPart1 />
            </TabsContent>

            <TabsContent value="optimization2" className="mt-6">
              <h2 className="text-2xl font-bold mb-4 text-green-400">CUDA Optimization - Part 2</h2>
              <p className="text-gray-300 mb-6">
                Learn advanced optimization techniques including memory throughput, aligned access, and bank conflict
                avoidance.
              </p>
              <CudaOptimizationPart2 />
            </TabsContent>

            <TabsContent value="atomics" className="mt-6">
              <h2 className="text-2xl font-bold mb-4 text-green-400">Atomics in CUDA</h2>
              <p className="text-gray-300 mb-6">
                Master atomic operations for thread-safe memory updates, parallel reductions, and synchronization
                primitives.
              </p>
              <CudaAtomicsTutorial />
            </TabsContent>

            <TabsContent value="compilation" className="mt-6">
              <h2 className="text-2xl font-bold mb-4 text-green-400">CUDA Compilation and Execution</h2>
              <p className="text-gray-300 mb-6">
                Learn how to compile, run, and profile CUDA applications using the NVIDIA CUDA Compiler (nvcc) and
                performance analysis tools.
              </p>
              <CudaCompilerGuide />
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* Code Example Section */}
      <section className="py-12 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-gray-800 rounded-lg shadow-lg overflow-hidden">
            <div className="px-6 py-4 bg-gray-700">
              <div className="flex justify-between items-center">
                <h3 className="text-xl font-semibold text-green-400">Optimizing Performance</h3>
                <span className="text-sm text-gray-300">Step 5 of 5</span>
              </div>
              <p className="text-gray-300 text-sm">Learn advanced techniques for optimizing CUDA code</p>
            </div>
            <div className="p-6 bg-gray-900 overflow-x-auto">
              <pre className="text-gray-300 text-sm">
                <code>
                  {`#include <stdio.h>
#include <cuda_runtime.h>

// Helper function for error checking
#define checkCudaErrors(call) {  cudaError_t err = call;  if (err != cudaSuccess) {  fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err));  exit(1); }}

// Optimized CUDA kernel for vector addition
__global__ void vectorAddOptimized(float *a, float *b, float *c, int n) {
// Get global thread ID
int i = blockIdx.x * blockDim.x + threadIdx.x;

// Each thread processes multiple elements (loop unrolling)
const int elementsPerThread = 4;
const int stride = blockDim.x * gridDim.x;

for (int idx = i; idx < n; idx += stride) {
  // Process multiple elements per thread
  for (int j = 0; j < elementsPerThread && idx + j * stride < n; j++) {
    int index = idx + j * stride;
    c[index] = a[index] + b[index];
  }
}
}`}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Resources Section */}
      <section id="resources" className="py-12 bg-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-extrabold text-green-400 sm:text-4xl">CUDA Resources</h2>
            <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-300">
              Explore these official NVIDIA resources to deepen your understanding of CUDA programming
            </p>
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            {/* CUDA Zone */}
            <div className="bg-gray-900 rounded-lg shadow-lg overflow-hidden">
              <a href="https://developer.nvidia.com/cuda-zone" target="_blank" rel="noopener noreferrer">
                <img
                  src="https://sjc.microlink.io/6NPrACaXpxaD2B3fb5WndDSLfE5YHkoWvfxcDW7WhwBsCM13zLCB8YhTbfaW5fXpiOdoTs5km4wQbTtyoyBQ_Q.jpeg"
                  alt="NVIDIA CUDA Zone"
                  className="w-full h-48 object-cover object-top"
                />
                <div className="px-6 py-4">
                  <h3 className="text-xl font-semibold text-green-400">CUDA Zone</h3>
                  <p className="mt-2 text-gray-300">
                    The official NVIDIA CUDA Zone provides an overview of CUDA technology, applications, and resources
                    for developers.
                  </p>
                </div>
              </a>
            </div>

            {/* CUDA Toolkit */}
            <div className="bg-gray-900 rounded-lg shadow-lg overflow-hidden">
              <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noopener noreferrer">
                <img
                  src="https://sjc.microlink.io/IILV5frtDD0qgj8nUb_wcHQa9nsxsrTPgVS9LQQvhLsUScNPV7KH9KEv8CtxmgpRrbOwICdrSadKSILTVW7vdg.jpeg"
                  alt="NVIDIA CUDA Toolkit"
                  className="w-full h-48 object-cover object-top"
                />
                <div className="px-6 py-4">
                  <h3 className="text-xl font-semibold text-green-400">CUDA Toolkit</h3>
                  <p className="mt-2 text-gray-300">
                    Download the CUDA Toolkit which includes GPU-accelerated libraries, a compiler, development tools,
                    and the CUDA runtime.
                  </p>
                </div>
              </a>
            </div>

            {/* CUDA Video Tutorial */}
            <div className="bg-gray-900 rounded-lg shadow-lg overflow-hidden">
              <a
                href="https://www.google.com/search?q=cuda&rlz=1C1GCEU_enJO1129JO1129&oq=cuda&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MggIARBFGCcYOzIHCAIQABiABDIHCAMQABiABDIGCAQQRRg8MgYIBRBFGDwyBggGEEUYPDIGCAcQRRg80gEIMTQ0N2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:a15d0016,vid:GmNkYayuaA4,st:0"
                target="_blank"
                rel="noopener noreferrer"
              >
                <img
                  src="https://sjc.microlink.io/N1uqhNScr8JD2p8JAFsdwi2Ob5bA68u8TD7hWv60J6s1r8-W1LQabeokA0paJZFoWgL5wxJRADZW-FTv6fEP0w.jpeg"
                  alt="CUDA Search Results"
                  className="w-full h-48 object-cover object-top"
                />
                <div className="px-6 py-4">
                  <h3 className="text-xl font-semibold text-green-400">CUDA Video Tutorial</h3>
                  <p className="mt-2 text-gray-300">
                    Watch this comprehensive video tutorial to learn about CUDA programming concepts and implementation.
                  </p>
                </div>
              </a>
            </div>
          </div>

          <div className="mt-12 text-center">
            <h3 className="text-xl font-semibold text-green-400 mb-4">Additional Resources</h3>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
              <a
                href="https://docs.nvidia.com/cuda/"
                target="_blank"
                rel="noopener noreferrer"
                className="bg-gray-900 p-4 rounded-lg hover:bg-gray-700 transition-colors"
              >
                <h4 className="font-medium text-white">CUDA Documentation</h4>
              </a>
              <a
                href="https://developer.nvidia.com/cuda-education-resources"
                target="_blank"
                rel="noopener noreferrer"
                className="bg-gray-900 p-4 rounded-lg hover:bg-gray-700 transition-colors"
              >
                <h4 className="font-medium text-white">Educational Resources</h4>
              </a>
              <a
                href="https://developer.nvidia.com/blog/tag/cuda/"
                target="_blank"
                rel="noopener noreferrer"
                className="bg-gray-900 p-4 rounded-lg hover:bg-gray-700 transition-colors"
              >
                <h4 className="font-medium text-white">CUDA Blog Posts</h4>
              </a>
              <a
                href="https://forums.developer.nvidia.com/c/accelerated-computing/cuda/164"
                target="_blank"
                rel="noopener noreferrer"
                className="bg-gray-900 p-4 rounded-lg hover:bg-gray-700 transition-colors"
              >
                <h4 className="font-medium text-white">CUDA Forums</h4>
              </a>
            </div>
          </div>
        </div>
      </section>
      {/* Footer */}
      <footer className="bg-gray-800 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="md:flex md:items-center md:justify-between">
            <div className="flex justify-center md:order-2">
              <a href="#" className="text-gray-400 hover:text-gray-300">
                <span className="sr-only">GitHub</span>
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    fillRule="evenodd"
                    d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                  />
                </svg>
              </a>
            </div>
            <div className="mt-8 md:mt-0 md:order-1">
              <p className="text-center text-gray-400 text-sm">
                &copy; {new Date().getFullYear()} CUDA Parallelism. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      </footer>

      {/* The chatbot will appear as a floating widget in the bottom-right corner */}
      <CudaChatbot />
    </div>
  )
}
