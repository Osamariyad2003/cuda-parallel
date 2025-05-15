"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Copy, Check } from "lucide-react"

const cudaAtomicsTutorial = [
  {
    id: "intro",
    title: "Introduction to CUDA Atomics",
    content: `
# Introduction to CUDA Atomics

Atomic operations in CUDA allow threads to perform read-modify-write operations on memory without interference from other threads.

## What are Atomic Operations?

Atomic operations are indivisible - they complete in a single, uninterrupted step from the perspective of other threads. This ensures that when multiple threads attempt to modify the same memory location simultaneously, the operations are serialized and produce a consistent result.

## Why Use Atomics?

1. **Thread Safety**: Safely update shared variables from multiple threads
2. **Race Condition Prevention**: Avoid data corruption when multiple threads access the same memory
3. **Synchronization**: Implement synchronization primitives and algorithms
4. **Reduction Operations**: Perform parallel reductions without explicit synchronization

## Common Atomic Operations in CUDA

CUDA provides several atomic functions for different data types and operations:

- **atomicAdd**: Add a value to a memory location
- **atomicSub**: Subtract a value from a memory location
- **atomicExch**: Exchange a value with a memory location
- **atomicMin/atomicMax**: Store minimum/maximum of a value and a memory location
- **atomicAnd/atomicOr/atomicXor**: Perform bitwise operations
- **atomicCAS**: Compare and swap (fundamental atomic operation)

## Performance Considerations

Atomic operations have performance implications:

1. **Serialization**: Atomic operations on the same address are serialized
2. **Contention**: High contention (many threads accessing the same location) reduces performance
3. **Architecture Dependence**: Performance varies across GPU architectures
4. **Memory Type**: Atomics on shared memory are faster than on global memory

## When to Use Atomics

- When multiple threads need to update the same memory location
- For parallel reduction algorithms
- For implementing synchronization primitives
- For histogram and sparse matrix operations

## When to Avoid Atomics

- When alternative algorithms exist that don't require atomics
- When contention would be high (many threads accessing the same location)
- When performance is critical and atomic operations would be a bottleneck
- When the overhead of atomics outweighs the benefits
    `,
    code: `#include <stdio.h>

// Kernel demonstrating atomic operations
__global__ void atomicOperationsDemo(int* counter, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Atomic increment
        atomicAdd(&counter[0], 1);
        
        // Atomic addition with a value
        atomicAdd(&counter[1], idx);
        
        // Atomic maximum
        atomicMax(&counter[2], idx);
        
        // Atomic minimum
        atomicMin(&counter[3], idx);
        
        // Atomic exchange (returns old value)
        int old = atomicExch(&counter[4], idx);
        
        // Atomic compare-and-swap
        // If counter[5] equals 0, replace it with idx
        atomicCAS(&counter[5], 0, idx);
        
        // Store the result of one atomic operation
        if (idx == 0) {
            result[0] = old;
        }
    }
}

// Kernel demonstrating race conditions without atomics
__global__ void raceConditionDemo(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Non-atomic increment - will cause race conditions
        counter[0]++;
        
        // Non-atomic addition - will cause race conditions
        counter[1] += idx;
    }
}

int main() {
    int n = 1000;
    int numCounters = 6;
    
    // Allocate host memory
    int* h_counter = (int*)malloc(numCounters * sizeof(int));
    int* h_result = (int*)malloc(sizeof(int));
    
    // Initialize counters
    for (int i = 0; i < numCounters; i++) {
        h_counter[i] = 0;
    }
    
    // Allocate device memory
    int* d_counter;
    int* d_result;
    cudaMalloc(&d_counter, numCounters * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_counter, h_counter, numCounters * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with atomic operations
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    atomicOperationsDemo<<<gridSize, blockSize>>>(d_counter, d_result, n);
    
    // Copy results back to host
    cudaMemcpy(h_counter, d_counter, numCounters * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Atomic Operations Results:\\n");
    printf("Counter[0] (atomicAdd with 1): %d (expected %d)\\n", h_counter[0], n);
    printf("Counter[1] (atomicAdd with idx): %d\\n", h_counter[1]);
    printf("Counter[2] (atomicMax): %d (expected %d)\\n", h_counter[2], n-1);
    printf("Counter[3] (atomicMin): %d (expected 0)\\n", h_counter[3]);
    printf("Counter[4] (atomicExch): %d\\n", h_counter[4]);
    printf("Counter[5] (atomicCAS): %d\\n", h_counter[5]);
    printf("Result[0] (old value from atomicExch): %d\\n", h_result[0]);
    
    // Reset counters
    h_counter[0] = 0;
    h_counter[1] = 0;
    cudaMemcpy(d_counter, h_counter, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with race conditions
    raceConditionDemo<<<gridSize, blockSize>>>(d_counter, n);
    
    // Copy results back to host
    cudaMemcpy(h_counter, d_counter, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("\\nRace Condition Results (values will likely be incorrect):\\n");
    printf("Counter[0] (non-atomic increment): %d (expected %d)\\n", h_counter[0], n);
    printf("Counter[1] (non-atomic addition): %d\\n", h_counter[1]);
    
    // Cleanup
    cudaFree(d_counter);
    cudaFree(d_result);
    free(h_counter);
    free(h_result);
    
    return 0;
}`,
  },
  {
    id: "atomic-functions",
    title: "CUDA Atomic Functions",
    content: `
# CUDA Atomic Functions

CUDA provides a variety of atomic functions for different data types and operations. These functions are essential for thread-safe updates to shared memory locations.

## Basic Atomic Functions

### Integer Atomics (32-bit)

- **atomicAdd(address, val)**: Adds val to the value at address
- **atomicSub(address, val)**: Subtracts val from the value at address
- **atomicExch(address, val)**: Exchanges val with the value at address
- **atomicMin(address, val)**: Stores minimum of val and the value at address
- **atomicMax(address, val)**: Stores maximum of val and the value at address
- **atomicInc(address, val)**: Increments the value at address, wrapping at val
- **atomicDec(address, val)**: Decrements the value at address, wrapping at val
- **atomicCAS(address, compare, val)**: Compare and swap

### Bitwise Atomic Operations

- **atomicAnd(address, val)**: Performs bitwise AND
- **atomicOr(address, val)**: Performs bitwise OR
- **atomicXor(address, val)**: Performs bitwise XOR

## Extended Atomic Support

### 64-bit Atomics (Compute Capability 3.5+)

- **atomicAdd(address, val)** for unsigned long long int
- **atomicExch(address, val)** for unsigned long long int
- **atomicCAS(address, compare, val)** for unsigned long long int

### Floating-Point Atomics

- **atomicAdd(address, val)** for float (all devices)
- **atomicAdd(address, val)** for double (Compute Capability 6.0+)
- **atomicExch(address, val)** for float

## Memory Spaces

Atomic functions can operate on different memory spaces:

- **Global Memory**: Accessible by all threads
- **Shared Memory**: Accessible by threads within a block
- **System Memory**: Through Unified Memory (Compute Capability 6.0+)

## Return Values

All atomic functions return the old value (before the operation) at the specified address. This can be useful for implementing more complex atomic operations.

## Implementing Custom Atomics

For operations not directly supported, you can implement custom atomic operations using atomicCAS (Compare And Swap):

1. Read the current value
2. Compute the new value
3. Use atomicCAS to update only if the value hasn't changed
4. Repeat if the update fails
    `,
    code: `#include <stdio.h>

// Custom atomic add for double precision (for devices before Compute Capability 6.0)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                         __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// Kernel demonstrating various atomic functions
__global__ void atomicFunctionsDemo(int* intResults, float* floatResults, double* doubleResults) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Integer atomic operations
    atomicAdd(&intResults[0], 1);                // Add 1
    atomicSub(&intResults[1], 1);                // Subtract 1
    atomicExch(&intResults[2], idx);             // Exchange
    atomicMin(&intResults[3], idx);              // Minimum
    atomicMax(&intResults[4], idx);              // Maximum
    atomicInc((unsigned int*)&intResults[5], 16); // Increment with wrap
    atomicDec((unsigned int*)&intResults[6], 16); // Decrement with wrap
    atomicAnd(&intResults[7], 0xF);              // Bitwise AND
    atomicOr(&intResults[8], 0xF0);              // Bitwise OR
    atomicXor(&intResults[9], 0xFF);             // Bitwise XOR
    
    // Float atomic operations
    atomicAdd(&floatResults[0], 1.0f);           // Add 1.0
    
    // Double atomic operations (using custom implementation)
    atomicAddDouble(&doubleResults[0], 1.0);     // Add 1.0
    
    // Using atomicCAS for custom operation (increment by 2)
    int old, assumed;
    do {
        old = intResults[10];
        assumed = old;
        old = atomicCAS(&intResults[10], assumed, assumed + 2);
    } while (assumed != old);
}

// Kernel demonstrating atomic operations in shared memory
__global__ void sharedMemoryAtomicsDemo(int* results) {
    __shared__ int sharedCounter;
    __shared__ int sharedMax;
    
    // Initialize shared memory (only one thread per block)
    if (threadIdx.x == 0) {
        sharedCounter = 0;
        sharedMax = 0;
    }
    
    // Ensure shared memory is initialized
    __syncthreads();
    
    // Perform atomic operations on shared memory
    atomicAdd(&sharedCounter, 1);
    atomicMax(&sharedMax, threadIdx.x);
    
    // Wait for all threads to complete
    __syncthreads();
    
    // Copy results to global memory (only one thread per block)
    if (threadIdx.x == 0) {
        results[blockIdx.x] = sharedCounter;
        results[blockIdx.x + gridDim.x] = sharedMax;
    }
}

int main() {
    int numBlocks = 32;
    int threadsPerBlock = 256;
    int numThreads = numBlocks * threadsPerBlock;
    
    // Allocate host memory
    int* h_intResults = (int*)malloc(11 * sizeof(int));
    float* h_floatResults = (float*)malloc(sizeof(float));
    double* h_doubleResults = (double*)malloc(sizeof(double));
    int* h_sharedResults = (int*)malloc(2 * numBlocks * sizeof(int));
    
    // Initialize results
    for (int i = 0; i < 11; i++) {
        h_intResults[i] = 0;
    }
    h_floatResults[0] = 0.0f;
    h_doubleResults[0] = 0.0;
    
    // Allocate device memory
    int* d_intResults;
    float* d_floatResults;
    double* d_doubleResults;
    int* d_sharedResults;
    
    cudaMalloc(&d_intResults, 11 * sizeof(int));
    cudaMalloc(&d_floatResults, sizeof(float));
    cudaMalloc(&d_doubleResults, sizeof(double));
    cudaMalloc(&d_sharedResults, 2 * numBlocks * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_intResults, h_intResults, 11 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_floatResults, h_floatResults, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_doubleResults, h_doubleResults, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sharedResults, h_sharedResults, 2 * numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel for global memory atomics
    atomicFunctionsDemo<<<numBlocks, threadsPerBlock>>>(d_intResults, d_floatResults, d_doubleResults);
    
    // Launch kernel for shared memory atomics
    sharedMemoryAtomicsDemo<<<numBlocks, threadsPerBlock>>>(d_sharedResults);
    
    // Copy results back to host
    cudaMemcpy(h_intResults, d_intResults, 11 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_floatResults, d_floatResults, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_doubleResults, d_doubleResults, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sharedResults, d_sharedResults, 2 * numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Global Memory Atomic Operations Results:\\n");
    printf("atomicAdd (int): %d (expected %d)\\n", h_intResults[0], numThreads);
    printf("atomicSub: %d (expected %d)\\n", h_intResults[1], -numThreads);
    printf("atomicExch: %d\\n", h_intResults[2]);
    printf("atomicMin: %d (expected 0)\\n", h_intResults[3]);
    printf("atomicMax: %d (expected %d)\\n", h_intResults[4], numThreads - 1);
    printf("atomicInc: %d\\n", h_intResults[5]);
    printf("atomicDec: %d\\n", h_intResults[6]);
    printf("atomicAnd: %d (expected 0)\\n", h_intResults[7]);
    printf("atomicOr: %d (expected 240)\\n", h_intResults[8]);
    printf("atomicXor: %d\\n", h_intResults[9]);
    printf("Custom CAS (increment by 2): %d (expected %d)\\n", h_intResults[10], numThreads * 2);
    printf("atomicAdd (float): %f (expected %f)\\n", h_floatResults[0], (float)numThreads);
    printf("atomicAdd (double): %lf (expected %lf)\\n", h_doubleResults[0], (double)numThreads);
    
    printf("\\nShared Memory Atomic Operations Results:\\n");
    printf("Block 0 - atomicAdd: %d (expected %d)\\n", h_sharedResults[0], threadsPerBlock);
    printf("Block 0 - atomicMax: %d (expected %d)\\n", h_sharedResults[numBlocks], threadsPerBlock - 1);
    
    // Cleanup
    cudaFree(d_intResults);
    cudaFree(d_floatResults);
    cudaFree(d_doubleResults);
    cudaFree(d_sharedResults);
    
    free(h_intResults);
    free(h_floatResults);
    free(h_doubleResults);
    free(h_sharedResults);
    
    return 0;
}`,
  },
  {
    id: "parallel-reduction",
    title: "Parallel Reduction with Atomics",
    content: `
# Parallel Reduction with Atomics

Parallel reduction is a common pattern in parallel computing for operations like sum, min, max, and other associative operations across an array. Atomics provide a simple way to implement reductions, though they may not always be the most efficient approach.

## Reduction Patterns

### 1. Naive Atomic Reduction

- Each thread processes one element and atomically updates a single global result
- Simple to implement but can suffer from high contention
- Works well for small data sizes or when atomics are fast

### 2. Two-Phase Reduction

- First phase: Each block computes a partial result using shared memory
- Second phase: One thread per block atomically updates the global result
- Reduces atomic contention significantly

### 3. Tree-Based Reduction

- Threads cooperatively reduce within a block using shared memory
- No atomics needed within blocks
- Only one atomic operation per block to update the global result

### 4. Warp-Level Reduction

- Uses warp shuffle operations for the first level of reduction
- Reduces shared memory usage and synchronization
- Combines with block-level and atomic global updates

## Performance Considerations

1. **Contention**: High contention on a single atomic variable reduces performance
2. **Work per Thread**: Increasing work per thread reduces the number of atomic operations
3. **Memory Coalescing**: Ensure coalesced memory access patterns for input data
4. **Shared Memory**: Use shared memory for intermediate reductions
5. **Warp Shuffles**: Use warp shuffle operations when possible (faster than shared memory)

## Applications of Atomic Reductions

- Computing sums, averages, min/max values
- Histograms and counting
- Sparse matrix operations
- Parallel graph algorithms
    `,
    code: `#include <stdio.h>

// Naive atomic reduction - each thread atomically adds to a global sum
__global__ void atomicReductionNaive(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

// Two-phase reduction - block reduction followed by atomic update
__global__ void atomicReductionTwoPhase(float* input, float* output, int n) {
    __shared__ float sharedSum;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        sharedSum = 0.0f;
    }

    __syncthreads();

    // Each thread atomically adds to the block's shared sum
    if (idx < n) {
        atomicAdd(&sharedSum, input[idx]);
    }

    __syncthreads();

    // One thread per block atomically adds the block's sum to the global sum
    if (threadIdx.x == 0) {
        atomicAdd(output, sharedSum);
    }
}

// Tree-based reduction within block, then atomic update
__global__ void atomicReductionTree(float* input, float* output, int n) {
    __shared__ float sharedData[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;

    __syncthreads();

    // Tree-based reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // One thread per block atomically adds the block's sum to the global sum
    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

// Warp-level reduction with shuffle, then atomic update
__global__ void atomicReductionWarpShuffle(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;  // Lane index within warp
    int warpId = threadIdx.x / 32; // Warp index within block
    int numWarps = (blockDim.x + 31) / 32; // Number of warps per block

    __shared__ float warpSums[8]; // Assuming at most 8 warps per block

    float sum = (idx < n) ? input[idx] : 0.0f;

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes the warp's sum
    if (lane == 0) {
        warpSums[warpId] = sum;
    }

    __syncthreads();

    // First warp reduces the warp sums
    if (warpId == 0 && lane < numWarps) {
        float warpSum = warpSums[lane];

        // Warp-level reduction of the warp sums
        for (int offset = 4; offset > 0; offset >>= 1) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }

        // First thread atomically adds the block's sum to the global sum
        if (lane == 0) {
            atomicAdd(output, warpSum);
        }
    }
}

// Histogram computation using atomics
__global__ void histogram(unsigned char* input, int* bins, int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        unsigned char value = input[idx];
        atomicAdd(&bins[value % numBins], 1);
    }
}

int main() {
    int n = 1000000;
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(sizeof(float));

    // Initialize input data
    float expectedSum = 0.0f;
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 10;
        expectedSum += h_input[i];
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test different reduction implementations
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 1. Naive atomic reduction
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);
    atomicReductionNaive<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naiveTime = 0;
    cudaEventElapsedTime(&naiveTime, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    float naiveSum = *h_output;

    printf("Naive Atomic Reduction: Sum = %f, Time = %f ms\\n", naiveSum, naiveTime);

    // 2. Two-phase atomic reduction
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);
    atomicReductionTwoPhase<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float twoPhaseTime = 0;
    cudaEventElapsedTime(&twoPhaseTime, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    float twoPhaseSum = *h_output;

    printf("Two-Phase Atomic Reduction: Sum = %f, Time = %f ms\\n", twoPhaseSum, twoPhaseTime);

    // 3. Tree-based atomic reduction
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);
    atomicReductionTree<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float treeTime = 0;
    cudaEventElapsedTime(&treeTime, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    float treeSum = *h_output;

    printf("Tree-Based Atomic Reduction: Sum = %f, Time = %f ms\\n", treeSum, treeTime);

    // 4. Warp-level reduction with shuffle
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);
    atomicReductionWarpShuffle<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float warpShuffleTime = 0;
    cudaEventElapsedTime(&warpShuffleTime, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    float warpShuffleSum = *h_output;

    printf("Warp-Level Shuffle Atomic Reduction: Sum = %f, Time = %f ms\\n", warpShuffleSum, warpShuffleTime);

    // Verify results
    printf("\\nExpected Sum: %f\\n", expectedSum);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}`,
  },
]

export default function CudaAtomicsTutorial() {
  const [currentStep, setCurrentStep] = useState(0)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(cudaAtomicsTutorial[currentStep].code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const nextStep = () => {
    if (currentStep < cudaAtomicsTutorial.length - 1) {
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
              <CardTitle className="text-green-400">{cudaAtomicsTutorial[currentStep].title}</CardTitle>
              <CardDescription>Atomics in CUDA</CardDescription>
            </div>
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {cudaAtomicsTutorial.length}
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
                  <div
                    dangerouslySetInnerHTML={{
                      __html: cudaAtomicsTutorial[currentStep].content
                        .replace(/\n/g, "<br>")
                        .replace(/# (.*)/g, "<h1>$1</h1>")
                        .replace(/## (.*)/g, "<h2>$1</h2>")
                        .replace(/### (.*)/g, "<h3>$1</h3>")
                        .replace(/\*\*(.*)\*\*/g, "<strong>$1</strong>")
                        .replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
                        .replace(/`(.*?)`/g, "<code>$1</code>"),
                    }}
                  />
                </div>
              </div>
            </TabsContent>
            <TabsContent value="code" className="mt-4">
              <div className="relative">
                <pre className="bg-gray-900 p-4 rounded-md overflow-x-auto text-sm text-gray-300 max-h-[500px] overflow-y-auto">
                  {cudaAtomicsTutorial[currentStep].code}
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
            disabled={currentStep === cudaAtomicsTutorial.length - 1}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
