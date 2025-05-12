"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, ExternalLink, Copy, Check } from "lucide-react"

const tutorialSteps = [
  {
    title: "Setting Up CUDA",
    description: "Install the CUDA Toolkit and set up your development environment",
    code: `// 1. Download and install the CUDA Toolkit from NVIDIA's website
// 2. Verify installation with:
#include <stdio.h>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Number of CUDA devices: %d\\n", deviceCount);
  
  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("Device %d: %s\\n", i, deviceProp.name);
    printf("  Compute Capability: %d.%d\\n", deviceProp.major, deviceProp.minor);
    printf("  Total Global Memory: %.2f GB\\n", 
           deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }
  
  return 0;
}`,
    explanation:
      "This code checks if CUDA is properly installed by querying the available CUDA devices and their properties. It's a good first test to ensure your setup is working correctly.",
  },
  {
    title: "Your First CUDA Kernel",
    description: "Write a simple vector addition kernel",
    code: `#include <stdio.h>

// CUDA Kernel for vector addition
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
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
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
  
  // Verify result
  for (int i = 0; i < n; i++) {
    if (h_c[i] != 3.0f) {
      printf("Error: Result verification failed at element %d!\\n", i);
      break;
    }
  }
  
  printf("Vector addition completed successfully!\\n");
  
  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  // Free host memory
  free(h_a);
  free(h_b);
  free(h_c);
  
  return 0;
}`,
    explanation:
      "This example demonstrates the basic structure of a CUDA program. The __global__ keyword indicates a function that runs on the GPU (a kernel). The kernel is launched with a specific grid and block configuration, which determines how many threads will execute the kernel in parallel.",
  },
  {
    title: "Understanding Thread Hierarchy",
    description: "Learn about CUDA's thread, block, and grid organization",
    code: `#include <stdio.h>

// CUDA Kernel to demonstrate thread hierarchy
__global__ void threadHierarchy() {
  // Thread index within a block
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  // Block index within a grid
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  
  // Block dimensions
  int bdx = blockDim.x;
  int bdy = blockDim.y;
  int bdz = blockDim.z;
  
  // Grid dimensions
  int gdx = gridDim.x;
  int gdy = gridDim.y;
  int gdz = gridDim.z;
  
  // Calculate global thread ID
  int globalIdx = bx * bdx + tx;
  int globalIdy = by * bdy + ty;
  int globalIdz = bz * bdz + tz;
  
  // Print information for the first thread only to avoid clutter
  if (tx == 0 && ty == 0 && tz == 0 && bx == 0 && by == 0 && bz == 0) {
    printf("Grid dimensions: (%d, %d, %d)\\n", gdx, gdy, gdz);
    printf("Block dimensions: (%d, %d, %d)\\n", bdx, bdy, bdz);
    printf("Total threads: %d\\n", gdx * gdy * gdz * bdx * bdy * bdz);
  }
}

int main() {
  // Define grid and block dimensions
  dim3 blockDim(8, 8, 1);  // 8x8x1 = 64 threads per block
  dim3 gridDim(4, 4, 1);   // 4x4x1 = 16 blocks
  
  // Launch kernel
  threadHierarchy<<<gridDim, blockDim>>>();
  
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  
  return 0;
}`,
    explanation:
      "This example demonstrates CUDA's thread hierarchy. Threads are organized into blocks, and blocks are organized into a grid. Each thread has a unique ID within its block (threadIdx), and each block has a unique ID within the grid (blockIdx). This hierarchical organization allows CUDA to scale from small GPUs with a few cores to large GPUs with thousands of cores.",
  },
  {
    title: "Using Shared Memory",
    description: "Optimize performance with shared memory",
    code: `#include <stdio.h>

// CUDA Kernel for matrix multiplication using shared memory
__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // Define shared memory tiles
  __shared__ float As[16][16];
  __shared__ float Bs[16][16];
  
  // Calculate row and column indices
  int row = by * 16 + ty;
  int col = bx * 16 + tx;
  
  float sum = 0.0f;
  
  // Loop over tiles
  for (int tile = 0; tile < (N + 15) / 16; tile++) {
    // Load data into shared memory
    if (row < N && tile * 16 + tx < N) {
      As[ty][tx] = A[row * N + tile * 16 + tx];
    } else {
      As[ty][tx] = 0.0f;
    }
    
    if (col < N && tile * 16 + ty < N) {
      Bs[ty][tx] = B[(tile * 16 + ty) * N + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Compute partial dot product
    for (int k = 0; k < 16; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }
    
    // Synchronize before loading next tile
    __syncthreads();
  }
  
  // Write result
  if (row < N && col < N) {
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
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
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
  matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  
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
}`,
    explanation:
      "This example demonstrates the use of shared memory in CUDA. Shared memory is a fast, on-chip memory that can be accessed by all threads in a block. By loading data into shared memory, we can reduce the number of global memory accesses, which are much slower. The __syncthreads() function is used to ensure all threads in a block have finished loading data into shared memory before proceeding.",
  },
  {
    title: "Optimizing Performance",
    description: "Learn advanced techniques for optimizing CUDA code",
    code: `#include <stdio.h>
#include <cuda_runtime.h>

// Helper function for error checking
#define checkCudaErrors(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in %s at line %d: %s\\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

// Optimized CUDA Kernel for vector addition
__global__ void vectorAddOptimized(float *a, float *b, float *c, int n) {
  // Get global thread ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Each thread processes multiple elements (loop unrolling)
  const int elementsPerThread = 4;
  const int stride = blockDim.x * gridDim.x;
  
  for (int idx = i; idx < n; idx += stride * elementsPerThread) {
    // Process multiple elements per thread if in bounds
    if (idx < n) c[idx] = a[idx] + b[idx];
    if (idx + stride < n) c[idx + stride] = a[idx + stride] + b[idx + stride];
    if (idx + 2 * stride < n) c[idx + 2 * stride] = a[idx + 2 * stride] + b[idx + 2 * stride];
    if (idx + 3 * stride < n) c[idx + 3 * stride] = a[idx + 3 * stride] + b[idx + 3 * stride];
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
  
  // Allocate page-locked host memory for better transfer speeds
  checkCudaErrors(cudaMallocHost(&h_a, bytes));
  checkCudaErrors(cudaMallocHost(&h_b, bytes));
  checkCudaErrors(cudaMallocHost(&h_c, bytes));
  
  // Initialize vectors
  for (int i = 0; i < n; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }
  
  // Allocate device memory
  checkCudaErrors(cudaMalloc(&d_a, bytes));
  checkCudaErrors(cudaMalloc(&d_b, bytes));
  checkCudaErrors(cudaMalloc(&d_c, bytes));
  
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  
  // Start timing
  checkCudaErrors(cudaEventRecord(start));
  
  // Use streams for asynchronous operations
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  
  // Copy data from host to device asynchronously
  checkCudaErrors(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
  
  // Set grid and block dimensions
  int blockSize = 256;
  int gridSize = (n / 4 + blockSize - 1) / blockSize; // Adjusted for 4 elements per thread
  
  // Launch kernel
  vectorAddOptimized<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, n);
  
  // Copy result back to host asynchronously
  checkCudaErrors(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
  
  // Wait for all operations to complete
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  // Stop timing
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  
  // Calculate elapsed time
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Execution time: %.3f ms\\n", milliseconds);
  
  // Verify result
  bool success = true;
  for (int i = 0; i < n; i++) {
    if (fabs(h_c[i] - 3.0f) > 1e-5) {
      printf("Error: Result verification failed at element %d!\\n", i);
      success = false;
      break;
    }
  }
  
  if (success) {
    printf("Vector addition completed successfully!\\n");
  }
  
  // Clean up
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
  
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));
  
  checkCudaErrors(cudaFreeHost(h_a));
  checkCudaErrors(cudaFreeHost(h_b));
  checkCudaErrors(cudaFreeHost(h_c));
  
  return 0;
}`,
    explanation:
      "This example demonstrates several optimization techniques for CUDA: 1) Loop unrolling to process multiple elements per thread, 2) Using page-locked (pinned) memory for faster host-device transfers, 3) Using CUDA streams for asynchronous operations, 4) Error checking with a helper macro, and 5) Performance timing with CUDA events. These techniques can significantly improve the performance of your CUDA applications.",
  },
]

export default function CudaTutorial() {
  const [currentStep, setCurrentStep] = useState(0)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(tutorialSteps[currentStep].code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const nextStep = () => {
    if (currentStep < tutorialSteps.length - 1) {
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
              <CardTitle className="text-green-400">{tutorialSteps[currentStep].title}</CardTitle>
              <CardDescription>{tutorialSteps[currentStep].description}</CardDescription>
            </div>
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {tutorialSteps.length}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <pre className="bg-gray-900 p-4 rounded-md overflow-x-auto text-sm text-gray-300 max-h-[400px] overflow-y-auto">
              {tutorialSteps[currentStep].code}
            </pre>
            <Button
              variant="outline"
              size="sm"
              className="absolute top-2 right-2 bg-gray-800 border-gray-700 hover:bg-gray-700"
              onClick={handleCopy}
            >
              {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            </Button>
          </div>
          <div className="mt-4 p-4 bg-gray-900 rounded-md">
            <h4 className="text-white font-medium mb-2">Explanation</h4>
            <p className="text-gray-300 text-sm">{tutorialSteps[currentStep].explanation}</p>
          </div>
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
            className="border-green-500 text-green-500 hover:bg-green-500/10"
            onClick={() => window.open("https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html", "_blank")}
          >
            CUDA Documentation <ExternalLink className="ml-2 h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            className="border-gray-700 text-gray-300 hover:bg-gray-700"
            onClick={nextStep}
            disabled={currentStep === tutorialSteps.length - 1}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
