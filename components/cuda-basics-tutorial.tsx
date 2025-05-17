"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Copy, Check } from "lucide-react"

const cudaBasicsTutorial = [
  {
    id: "intro",
    title: "Introduction to CUDA and GPU Architecture",
    content: `
# Introduction to CUDA and GPU Architecture

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of GPUs.

## Key Concepts

- **GPU (Graphics Processing Unit)**: Originally designed for rendering graphics, now used for general-purpose computing
- **SIMT Architecture**: Single Instruction, Multiple Threads - many threads execute the same instruction simultaneously
- **Massive Parallelism**: GPUs contain thousands of cores optimized for parallel workloads
- **Heterogeneous Computing**: Using both CPU and GPU together, each for what they do best

## GPU Architecture Overview

GPUs are designed with a fundamentally different architecture than CPUs:

- **CPU**: Few cores (4-64) optimized for sequential processing with complex control logic
- **GPU**: Many simple cores (thousands) optimized for data-parallel processing

The GPU architecture consists of:

- **Streaming Multiprocessors (SMs)**: The main computational units
- **CUDA Cores**: Individual processing units within each SM
- **Memory Hierarchy**: Various types of memory with different performance characteristics
- **Warp Scheduler**: Manages groups of 32 threads (warps) that execute in lockstep
    `,
    code: `// No code for this introductory section
// CUDA programs are written in C/C++ with NVIDIA extensions`,
  },
  {
    id: "host-device",
    title: "Host vs. Device: CPU vs. GPU",
    content: `
# Host vs. Device: CPU vs. GPU

In CUDA programming, we distinguish between:

- **Host**: The CPU and its memory (system RAM)
- **Device**: The GPU and its memory (VRAM)

## Memory Spaces

1. **Host Memory**: Accessible only by the CPU
2. **Device Memory**: Accessible only by the GPU
3. **Unified Memory**: A memory model that creates a pool accessible by both (available in newer CUDA versions)

## Responsibilities

### Host (CPU) Responsibilities:
- Initialize data
- Transfer data to/from the device
- Launch kernels (GPU functions)
- Synchronize execution
- Handle sequential parts of the algorithm

### Device (GPU) Responsibilities:
- Execute parallel computations (kernels)
- Process large datasets in parallel
- Perform compute-intensive operations

## Programming Model

CUDA follows a heterogeneous programming model:
1. Allocate memory on the device
2. Copy input data from host to device
3. Execute kernel on the device
4. Copy results from device back to host
5. Free device memory
    `,
    code: `#include <stdio.h>

int main() {
    // Host code executes on the CPU
    printf("This code runs on the host (CPU)\\n");
    
    // Device code must be launched as a kernel
    // We'll see how to do this in the next sections
    
    return 0;
}`,
  },
  {
    id: "kernels",
    title: "Writing and Launching CUDA Kernels",
    content: `
# Writing and Launching CUDA Kernels

A **kernel** is a function that runs on the GPU. It's executed by many threads in parallel.

## Kernel Definition

- Defined using the \`__global__\` specifier
- Returns \`void\` (CUDA 11.0+ allows non-void returns)
- Executed on the device (GPU)
- Called from the host (CPU)

## Kernel Launch Syntax

Kernels are launched using the triple angle bracket syntax:

\`\`\`
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(parameters);
\`\`\`

Where:
- **gridDim**: Number of blocks in the grid
- **blockDim**: Number of threads in each block
- **sharedMemSize**: (Optional) Size of shared memory to allocate
- **stream**: (Optional) Stream to execute the kernel in

## Thread Organization

CUDA organizes threads in a hierarchical structure:
- **Thread**: Individual execution unit
- **Block**: Group of threads that can cooperate via shared memory
- **Grid**: Collection of blocks

This organization allows CUDA to scale from small GPUs with few cores to large GPUs with thousands of cores.
    `,
    code: `#include <stdio.h>

// Kernel definition
__global__ void helloFromGPU() {
    // This code runs on the GPU
    printf("Hello from thread [%d,%d] in block [%d,%d]\\n", 
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}

int main() {
    // Define grid and block dimensions
    dim3 blockDim(2, 2);  // 2x2 = 4 threads per block
    dim3 gridDim(2, 1);   // 2x1 = 2 blocks in the grid
    
    // Launch kernel with 8 total threads (2 blocks × 4 threads)
    helloFromGPU<<<gridDim, blockDim>>>();
    
    // Wait for GPU to finish before accessing results
    cudaDeviceSynchronize();
    
    return 0;
}`,
  },
  {
    id: "memory-management",
    title: "Memory Management: cudaMalloc, cudaMemcpy, cudaFree",
    content: `
# Memory Management in CUDA

CUDA provides functions to allocate, copy, and free memory on the device (GPU).

## Key Memory Management Functions

### 1. cudaMalloc
Allocates memory on the GPU.

\`\`\`c
cudaError_t cudaMalloc(void** devPtr, size_t size);
\`\`\`

### 2. cudaMemcpy
Copies data between host and device memory.

\`\`\`c
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
\`\`\`

The \`cudaMemcpyKind\` parameter can be:
- \`cudaMemcpyHostToDevice\`: Copy from CPU to GPU
- \`cudaMemcpyDeviceToHost\`: Copy from GPU to CPU
- \`cudaMemcpyDeviceToDevice\`: Copy within GPU memory

### 3. cudaFree
Frees memory on the GPU.

\`\`\`c
cudaError_t cudaFree(void* devPtr);
\`\`\`

## Error Handling

Always check the return values of CUDA functions:

\`\`\`c
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    printf("CUDA error: %s\\n", cudaGetErrorString(err));
    // Handle error...
}
\`\`\`

## Memory Management Best Practices

- Minimize host-device transfers (they're slow)
- Reuse allocated memory when possible
- Free all allocated memory to avoid leaks
- Consider using Unified Memory for simpler code (\`cudaMallocManaged\`)
    `,
    code: `#include <stdio.h>

int main() {
    int N = 10;
    size_t size = N * sizeof(int);
    
    // Host memory
    int* h_data = (int*)malloc(size);
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    
    // Device memory
    int* d_data = NULL;
    
    // Allocate memory on the device
    cudaError_t err = cudaMalloc((void**)&d_data, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy data from host to device
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy (H2D) failed: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // At this point, we would launch a kernel to process d_data
    // ...
    
    // Copy results back from device to host
    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy (D2H) failed: %s\\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // Free device memory
    cudaFree(d_data);
    
    // Free host memory
    free(h_data);
    
    return 0;
}`,
  },
  {
    id: "vector-addition",
    title: "Vector Addition Example",
    content: `
# Vector Addition Example

Let's implement a simple vector addition example that demonstrates the complete CUDA workflow:

1. Allocate host memory and initialize data
2. Allocate device memory
3. Copy data from host to device
4. Launch kernel to perform computation
5. Copy results back from device to host
6. Verify results and clean up

## Vector Addition Kernel

The kernel function adds corresponding elements of two input arrays and stores the result in an output array. Each thread handles one element of the arrays.

## Thread Indexing

To calculate the global index of a thread, we use:

\`\`\`
int idx = blockIdx.x * blockDim.x + threadIdx.x;
\`\`\`

Where:
- \`blockIdx.x\`: Block index in the x-dimension
- \`blockDim.x\`: Number of threads per block in the x-dimension
- \`threadIdx.x\`: Thread index within the block in the x-dimension

This formula gives each thread a unique global ID that we use to determine which data element it should process.
    `,
    code: `#include <stdio.h>

// Vector addition kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Vector size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < numElements; i++) {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }
    
    // Allocate device memory
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch the Vector Add kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\\n", i);
            exit(1);
        }
    }
    printf("Test PASSED\\n");
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\\n");
    return 0;
}`,
  },
  {
    id: "thread-indexing",
    title: "Thread Indexing: threadIdx, blockIdx, blockDim",
    content: `
# Thread Indexing in CUDA

CUDA provides built-in variables to identify threads and blocks:

## Thread Identification Variables

- **threadIdx**: Thread index within a block (can be 1D, 2D, or 3D)
  - \`threadIdx.x\`, \`threadIdx.y\`, \`threadIdx.z\`

- **blockIdx**: Block index within the grid (can be 1D, 2D, or 3D)
  - \`blockIdx.x\`, \`blockIdx.y\`, \`blockIdx.z\`

- **blockDim**: Dimensions of a block (number of threads in each dimension)
  - \`blockDim.x\`, \`blockDim.y\`, \`blockDim.z\`

- **gridDim**: Dimensions of the grid (number of blocks in each dimension)
  - \`gridDim.x\`, \`gridDim.y\`, \`gridDim.z\`

## Calculating Global Thread IDs

### 1D Grid of 1D Blocks:
\`\`\`
int idx = blockIdx.x * blockDim.x + threadIdx.x;
\`\`\`

### 2D Grid of 2D Blocks:
\`\`\`
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = idx_y * width + idx_x;  // For a row-major layout
\`\`\`

### 3D Grid of 3D Blocks:
\`\`\`
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = idx_z * width * height + idx_y * width + idx_x;  // For a row-major layout
\`\`\`

## Choosing Thread Organization

The choice of thread organization depends on your problem:

- **1D**: Good for vector operations (e.g., vector addition)
- **2D**: Good for matrix operations and image processing
- **3D**: Good for volume processing (e.g., 3D simulations)
    `,
    code: `#include <stdio.h>

// Kernel demonstrating thread indexing
__global__ void threadIndexing() {
    // 1D indexing
    int idx_1d = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2D indexing (for a row-major layout)
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;  // Total width of the 2D grid
    int idx_2d = idx_y * width + idx_x;
    
    // Print thread information
    printf("Thread [%d,%d] in Block [%d,%d] - 1D index: %d, 2D index: %d\\n",
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx_1d, idx_2d);
}

int main() {
    // Define grid and block dimensions
    dim3 blockDim(4, 2);  // 4x2 = 8 threads per block
    dim3 gridDim(2, 2);   // 2x2 = 4 blocks in the grid
    
    // Launch kernel
    threadIndexing<<<gridDim, blockDim>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}`,
  },
]

export default function CudaBasicsTutorial() {
  const [currentStep, setCurrentStep] = useState(0)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(cudaBasicsTutorial[currentStep].code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const nextStep = () => {
    if (currentStep < cudaBasicsTutorial.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  return (
    <div className="w-full">
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-green-400">{cudaBasicsTutorial[currentStep].title}</CardTitle>
              <CardDescription>CUDA C++ Basics</CardDescription>
            </div>
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {cudaBasicsTutorial.length}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="content">
            <TabsList className="grid w-full grid-cols-2 bg-gray-900">
              <TabsTrigger value="content">Content</TabsTrigger>
              <TabsTrigger value="code">Code Example</TabsTrigger>
            </TabsList>
            <TabsContent value="content" className="mt-4">
              <div className="bg-gray-900 p-4 rounded-md overflow-y-auto max-h-[500px]">
                <div className="prose prose-invert max-w-none">
                  <div className="text-gray-200"
                    dangerouslySetInnerHTML={{
                      __html: cudaBasicsTutorial[currentStep].content
                        .replace(/\n/g, "<br>")
                        .replace(/# (.*?)(\n|$)/g, "<h1 class='text-2xl font-bold text-green-400 mb-4'>$1</h1>")
                        .replace(/## (.*?)(\n|$)/g, "<h2 class='text-xl font-semibold text-green-300 mt-6 mb-3'>$1</h2>")
                        .replace(/### (.*?)(\n|$)/g, "<h3 class='text-lg font-medium text-green-200 mt-4 mb-2'>$1</h3>")
                        .replace(/\*\*(.*?)\*\*/g, "<strong class='text-green-300'>$1</strong>")
                        .replace(/- (.*?)(\n|$)/g, "<li class='ml-4 text-gray-300'>$1</li>")
                        .replace(/```([\s\S]*?)```/g, "<pre class='bg-gray-800 p-3 rounded-md my-3 overflow-x-auto'><code>$1</code></pre>")
                        .replace(/`([^`]+)`/g, "<code class='bg-gray-800 px-1 rounded text-green-300'>$1</code>")
                  }}
                  />
                </div>
              </div>
            </TabsContent>
            <TabsContent value="code" className="mt-4">
              <div className="relative">
                <pre className="bg-gray-800 text-gray-200 p-4 rounded-lg overflow-x-auto font-mono text-sm">
                  <code>{cudaBasicsTutorial[currentStep].code}</code>
                </pre>
                <Button
                  variant="outline"
                  size="sm"
                  className="absolute top-2 right-2 bg-gray-700 border-gray-600 hover:bg-gray-600 text-gray-200"
                  onClick={handleCopy}
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button
            variant="outline"
            className="border-gray-700 text-gray-300 hover:bg-gray-700"
            onClick={prevStep}
            disabled={currentStep === 0}
          >
            <ChevronLeft className="mr-2 h-4 w-4" /> Previous
          </Button>
          <Button
            variant="outline"
            className="border-gray-700 text-gray-300 hover:bg-gray-700"
            onClick={nextStep}
            disabled={currentStep === cudaBasicsTutorial.length - 1}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
