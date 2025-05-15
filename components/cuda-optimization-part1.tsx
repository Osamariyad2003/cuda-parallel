"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Copy, Check } from "lucide-react"

const cudaOptimizationPart1 = [
  {
    id: "gpu-architecture",
    title: "GPU Architecture Evolution",
    content: `
# GPU Architecture Evolution

NVIDIA's GPU architectures have evolved significantly over time, with each generation bringing new features and performance improvements.

## Key NVIDIA GPU Architectures

### Kepler (2012-2014)
- Compute Capability: 3.0-3.7
- First architecture with Dynamic Parallelism
- Introduced Hyper-Q for multiple CPU cores to use the GPU simultaneously
- Key GPUs: Tesla K40, GeForce GTX 780

### Maxwell (2014-2016)
- Compute Capability: 5.0-5.3
- Improved power efficiency
- Enhanced shared memory and L2 cache
- Key GPUs: Tesla M40, GeForce GTX 980

### Pascal (2016-2017)
- Compute Capability: 6.0-6.2
- Introduced 16-bit floating point (FP16) operations
- Unified memory with page migration
- Key GPUs: Tesla P100, GeForce GTX 1080

### Volta (2017-2018)
- Compute Capability: 7.0
- Introduced Tensor Cores for accelerated deep learning
- Independent thread scheduling
- Key GPUs: Tesla V100, Titan V

### Turing (2018-2019)
- Compute Capability: 7.5
- Added RT Cores for ray tracing
- Enhanced Tensor Cores
- Key GPUs: Tesla T4, GeForce RTX 2080

### Ampere (2020-2022)
- Compute Capability: 8.0-8.6
- Third-generation Tensor Cores
- Sparsity acceleration
- Key GPUs: A100, GeForce RTX 3080

### Hopper (2022-present)
- Compute Capability: 9.0
- Fourth-generation Tensor Cores
- Transformer Engine
- Key GPUs: H100

### Ada Lovelace (2022-present)
- Compute Capability: 8.9
- Third-generation RT Cores
- DLSS 3.0
- Key GPUs: GeForce RTX 4090

## Architecture Impact on CUDA Programming

Each architecture introduces new features and optimizations that can be leveraged in CUDA code:

- **Compute Capability**: Determines available features and limitations
- **Memory Hierarchy**: Affects optimal memory access patterns
- **Warp Size**: Always 32 threads, but warp scheduling differs between architectures
- **Shared Memory**: Size and bank configuration varies by architecture
- **Register File**: Number of available registers per thread varies
    `,
    code: `// Check the compute capability of the device
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("Device %d: %s\\n", i, deviceProp.name);
        printf("  Compute Capability: %d.%d\\n", deviceProp.major, deviceProp.minor);
        printf("  Multiprocessors: %d\\n", deviceProp.multiProcessorCount);
        printf("  Warp Size: %d\\n", deviceProp.warpSize);
        printf("  Max Threads per Block: %d\\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max Blocks per Multiprocessor: %d\\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("  Shared Memory per Block: %zu bytes\\n", deviceProp.sharedMemPerBlock);
        printf("  Registers per Block: %d\\n", deviceProp.regsPerBlock);
        printf("  L2 Cache Size: %d bytes\\n", deviceProp.l2CacheSize);
        printf("  Memory Clock Rate: %d kHz\\n", deviceProp.memoryClockRate);
        printf("  Memory Bus Width: %d bits\\n", deviceProp.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\\n\\n", 
               2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
    }
    
    return 0;
}`,
  },
  {
    id: "memory-hierarchy",
    title: "Memory Hierarchy: Registers, Shared, Global, L1/L2 Cache",
    content: `
# Memory Hierarchy in CUDA

The CUDA memory hierarchy consists of several types of memory with different performance characteristics, scopes, and lifetimes.

## Memory Types (from fastest to slowest)

### 1. Registers
- **Speed**: Fastest (zero cycle latency for most operations)
- **Scope**: Per-thread
- **Lifetime**: Thread lifetime
- **Size**: Limited (typically 255 registers per thread)
- **Declaration**: Automatic variables in kernel functions
- **Use Case**: Local variables, frequently accessed data

### 2. Shared Memory
- **Speed**: Very fast (comparable to L1 cache)
- **Scope**: Per-block
- **Lifetime**: Block lifetime
- **Size**: Limited (typically 48KB-64KB per SM)
- **Declaration**: \`__shared__\` variables
- **Use Case**: Inter-thread communication, data reuse within a block

### 3. L1 Cache
- **Speed**: Fast
- **Scope**: Per-SM
- **Management**: Automatic
- **Size**: Typically 64KB-128KB per SM (often shared with shared memory)
- **Use Case**: Automatic caching of local and global memory

### 4. L2 Cache
- **Speed**: Medium
- **Scope**: Per-GPU
- **Management**: Automatic
- **Size**: Typically 512KB-6MB
- **Use Case**: Caching global memory accesses

### 5. Global Memory
- **Speed**: Slow (hundreds of cycles latency)
- **Scope**: All threads and host
- **Lifetime**: Application lifetime
- **Size**: Large (device VRAM, several GB)
- **Declaration**: \`cudaMalloc()\` or global variables with \`__device__\`
- **Use Case**: Main data storage, host-device communication

### 6. Constant Memory
- **Speed**: Fast when cached, slow otherwise
- **Scope**: All threads (read-only)
- **Size**: 64KB total
- **Declaration**: \`__constant__\` variables
- **Use Case**: Read-only data used by all threads

### 7. Texture Memory
- **Speed**: Medium (optimized for 2D/3D spatial locality)
- **Scope**: All threads (read-only)
- **Management**: Special texture cache
- **Use Case**: 2D/3D data with spatial locality

## Memory Optimization Strategies

1. **Maximize Register Usage**: Keep frequently accessed data in registers
2. **Use Shared Memory**: For data shared between threads in a block
3. **Coalesce Global Memory Accesses**: Adjacent threads should access adjacent memory
4. **Minimize Global Memory Accesses**: Use shared memory and registers as much as possible
5. **Use Read-Only Cache**: For data that won't change (via \`const __restrict__\`)
6. **Manage Occupancy**: Balance register usage with thread count
    `,
    code: `#include <stdio.h>

// Kernel demonstrating different memory types
__global__ void memoryTypesDemo(float* globalInput, float* globalOutput) {
    // Register memory (fastest, per-thread)
    float registerVar = 10.0f;
    
    // Shared memory (fast, per-block)
    __shared__ float sharedVar[256];
    
    // Calculate thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    sharedVar[threadIdx.x] = globalInput[idx];
    
    // Ensure all threads have initialized shared memory
    __syncthreads();
    
    // Use register and shared memory
    float result = registerVar * sharedVar[threadIdx.x];
    
    // Write result back to global memory
    globalOutput[idx] = result;
}

// Kernel demonstrating constant memory
__constant__ float constArray[256];

__global__ void constantMemoryDemo(float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Access constant memory (fast when cached)
    output[idx] = constArray[threadIdx.x % 256];
}

// Kernel demonstrating texture memory (simplified)
texture<float, 1, cudaReadModeElementType> texRef;

__global__ void textureMemoryDemo(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Access texture memory (optimized for spatial locality)
        output[idx] = tex1D(texRef, idx);
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    
    // Host memory
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_constData = (float*)malloc(256 * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }
    
    for (int i = 0; i < 256; i++) {
        h_constData[i] = i * 2.0f;
    }
    
    // Device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Copy data to constant memory
    cudaMemcpyToSymbol(constArray, h_constData, 256 * sizeof(float));
    
    // Set up texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTexture(NULL, texRef, d_input, channelDesc, bytes);
    
    // Launch kernels
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    memoryTypesDemo<<<gridSize, blockSize>>>(d_input, d_output);
    constantMemoryDemo<<<gridSize, blockSize>>>(d_output);
    textureMemoryDemo<<<gridSize, blockSize>>>(d_output, N);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaUnbindTexture(texRef);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_constData);
    
    return 0;
}`,
  },
  {
    id: "execution-model",
    title: "Execution Model: Warps, Thread Blocks, Grids",
    content: `
# CUDA Execution Model: Warps, Thread Blocks, Grids

The CUDA execution model is hierarchical, with threads organized into warps, blocks, and grids.

## Execution Hierarchy

### 1. Thread
- The basic unit of execution
- Executes a single instance of the kernel
- Has its own registers and local memory
- Identified by \`threadIdx\` within a block

### 2. Warp
- Group of 32 threads that execute in lockstep (SIMT)
- All threads in a warp execute the same instruction at the same time
- Divergence within a warp leads to serialization
- Not directly exposed in the programming model, but critical for performance

### 3. Thread Block
- Group of threads that can cooperate via shared memory
- Can synchronize using \`__syncthreads()\`
- Limited size (max 1024 threads on modern GPUs)
- Identified by \`blockIdx\` within a grid
- Scheduled to run on a single SM

### 4. Grid
- Collection of thread blocks
- Blocks in a grid execute independently
- Cannot directly synchronize between blocks
- Identified by \`gridDim\`

## Warp Execution

- **SIMT Architecture**: Single Instruction, Multiple Threads
- **Warp Size**: 32 threads on all current NVIDIA GPUs
- **Warp Scheduling**: SMs time-slice between warps
- **Warp Divergence**: When threads in a warp take different execution paths
- **Predication**: For short divergent paths, all paths are executed with predication
- **Branch Divergence**: For longer divergent paths, execution is serialized

## Thread Block Scheduling

- Blocks are scheduled to SMs as resources allow
- Once scheduled, a block runs to completion on that SM
- Multiple blocks can run concurrently on an SM if resources permit
- Blocks cannot depend on the execution order of other blocks

## Grid Execution

- A kernel launch creates a grid of thread blocks
- Blocks are distributed across SMs
- Grid execution completes when all blocks finish
- Multiple grids (kernels) can execute concurrently on newer architectures
    `,
    code: `#include <stdio.h>

// Kernel demonstrating warp execution
__global__ void warpExecutionDemo() {
    // Get thread ID within the warp (0-31)
    int laneId = threadIdx.x % 32;
    
    // Get warp ID within the block
    int warpId = threadIdx.x / 32;
    
    // Demonstrate warp-level operations
    
    // 1. Warp-level synchronization (implicit)
    int value = laneId;  // Each thread has its own value
    
    // 2. Warp divergence example
    if (laneId < 16) {
        // First half of the warp executes this
        value += 10;
    } else {
        // Second half of the warp executes this
        value *= 2;
    }
    
    // 3. Warp-level primitives (CUDA 9.0+)
    #if __CUDA_ARCH__ >= 900
    // Shuffle operations allow threads to exchange data within a warp
    int shuffled = __shfl_sync(0xffffffff, value, (laneId + 1) % 32);
    printf("Thread %d in warp %d: My value = %d, Neighbor's value = %d\\n", 
           laneId, warpId, value, shuffled);
    #else
    printf("Thread %d in warp %d: My value = %d\\n", laneId, warpId, value);
    #endif
}

// Kernel demonstrating block execution
__global__ void blockExecutionDemo() {
    // Shared memory visible to all threads in the block
    __shared__ int sharedData[256];
    
    // Each thread initializes its part of shared memory
    sharedData[threadIdx.x] = threadIdx.x;
    
    // Synchronize to ensure all threads have initialized shared memory
    __syncthreads();
    
    // Now all threads can safely read from shared memory
    int readValue = sharedData[(threadIdx.x + 1) % blockDim.x];
    
    // Only the first thread in each block prints
    if (threadIdx.x == 0) {
        printf("Block %d: First thread read value %d from shared memory\\n", 
               blockIdx.x, readValue);
    }
}

// Kernel demonstrating grid execution
__global__ void gridExecutionDemo() {
    // Calculate global thread ID
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only some threads print to avoid flooding the output
    if (globalId % 64 == 0) {
        printf("Grid execution: Thread %d in block %d (global ID: %d)\\n", 
               threadIdx.x, blockIdx.x, globalId);
    }
}

int main() {
    // Launch warp execution demo
    printf("=== Warp Execution Demo ===\\n");
    warpExecutionDemo<<<1, 64>>>();  // 2 warps
    cudaDeviceSynchronize();
    
    // Launch block execution demo
    printf("\\n=== Block Execution Demo ===\\n");
    blockExecutionDemo<<<4, 256>>>();  // 4 blocks, 256 threads each
    cudaDeviceSynchronize();
    
    // Launch grid execution demo
    printf("\\n=== Grid Execution Demo ===\\n");
    gridExecutionDemo<<<8, 256>>>();  // 8 blocks, 256 threads each
    cudaDeviceSynchronize();
    
    return 0;
}`,
  },
  {
    id: "launch-config",
    title: "Launch Configuration: Thread and Block Sizing",
    content: `
# Launch Configuration: Thread and Block Sizing

Choosing the right launch configuration (number of blocks and threads) is crucial for optimal performance.

## Key Considerations

### 1. Hardware Limits
- **Max Threads per Block**: Typically 1024
- **Max Blocks per Grid**: Typically 2³¹-1 in each dimension
- **Warp Size**: 32 threads
- **Max Warps per SM**: Architecture-dependent (typically 32-64)
- **Max Blocks per SM**: Architecture-dependent (typically 16-32)

### 2. Resource Constraints
- **Registers**: Limited per SM, shared among all threads
- **Shared Memory**: Limited per SM, allocated per block
- **Occupancy**: Percentage of maximum possible warps that can run concurrently

### 3. Problem Characteristics
- **Data Size**: Total number of elements to process
- **Data Access Pattern**: How threads access memory
- **Computation Intensity**: Ratio of computation to memory access
- **Thread Cooperation**: Need for shared memory and synchronization

## Thread Block Size Guidelines

1. **Multiple of 32 (Warp Size)**
   - Ensures full warp utilization
   - Common sizes: 128, 256, 512 threads

2. **Resource Considerations**
   - Larger blocks: Better shared memory efficiency, but fewer can run concurrently
   - Smaller blocks: More can run concurrently, but less shared memory efficiency

3. **Occupancy Targets**
   - 25-75% occupancy is often sufficient
   - 100% occupancy rarely needed and can be counterproductive

## Grid Size Calculation

Typically calculated to cover the entire data set:

\`\`\`cuda
int blockSize = 256;  // Threads per block
int gridSize = (n + blockSize - 1) / blockSize;  // Ceiling division
\`\`\`

## Occupancy Calculation

Occupancy is determined by:
1. Registers per thread
2. Shared memory per block
3. Block size

The CUDA Occupancy Calculator (or \`cudaOccupancyMaxPotentialBlockSize\`) can help determine optimal configurations.

## Dynamic Shared Memory

When using dynamic shared memory, include it in your launch configuration:

\`\`\`cuda
size_t sharedMemSize = blockSize * sizeof(float);
kernel<<<gridSize, blockSize, sharedMemSize>>>(args);
\`\`\`
    `,
    code: `#include <stdio.h>

// Helper function to calculate grid size
inline int calculateGridSize(int n, int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

// Kernel with different resource requirements
__global__ void resourceIntensiveKernel(float* input, float* output, int n) {
    // Use a lot of registers
    float a = 0.0f, b = 1.0f, c = 2.0f, d = 3.0f;
    float e = 4.0f, f = 5.0f, g = 6.0f, h = 7.0f;
    
    // Use shared memory
    extern __shared__ float sharedData[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Initialize shared memory
        sharedData[threadIdx.x] = input[idx];
        __syncthreads();
        
        // Perform computation using registers
        float result = a + b + c + d + e + f + g + h;
        
        // Use shared memory
        for (int i = 0; i < blockDim.x; i += 32) {
            if (threadIdx.x + i < blockDim.x) {
                result += sharedData[threadIdx.x + i];
            }
        }
        
        output[idx] = result;
    }
}

// Function to estimate occupancy
void estimateOccupancy(int blockSize) {
    int device;
    cudaDeviceProp prop;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate theoretical occupancy
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int warpsPerBlock = (blockSize + prop.warpSize - 1) / prop.warpSize;
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;
    
    if (maxBlocksPerSM > prop.maxBlocksPerMultiProcessor) {
        maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
    }
    
    int activeWarpsPerSM = warpsPerBlock * maxBlocksPerSM;
    int maxWarpsPerSM = maxThreadsPerSM / prop.warpSize;
    float occupancy = (float)activeWarpsPerSM / maxWarpsPerSM;
    
    printf("Block Size: %d\\n", blockSize);
    printf("Warps per Block: %d\\n", warpsPerBlock);
    printf("Max Blocks per SM: %d\\n", maxBlocksPerSM);
    printf("Active Warps per SM: %d\\n", activeWarpsPerSM);
    printf("Max Warps per SM: %d\\n", maxWarpsPerSM);
    printf("Theoretical Occupancy: %.2f%%\\n\\n", occupancy * 100);
}

int main() {
    // Data size
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    
    // Initialize input data
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 100;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Try different block sizes
    int blockSizes[] = {128, 256, 512, 1024};
    
    printf("Estimating occupancy for different block sizes:\\n\\n");
    for (int i = 0; i < 4; i++) {
        estimateOccupancy(blockSizes[i]);
    }
    
    // Launch kernel with different configurations
    printf("Launching kernel with different configurations:\\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < 4; i++) {
        int blockSize = blockSizes[i];
        int gridSize = calculateGridSize(n, blockSize);
        size_t sharedMemSize = blockSize * sizeof(float);
        
        // Measure execution time
        cudaEventRecord(start);
        resourceIntensiveKernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Block Size: %d, Grid Size: %d, Execution Time: %.3f ms\\n", 
               blockSize, gridSize, milliseconds);
    }
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}`,
  },
  {
    id: "occupancy",
    title: "Occupancy: Maximizing SM Utilization",
    content: `
# Occupancy: Maximizing SM Utilization

Occupancy is the ratio of active warps to the maximum number of warps supported on an SM.

## Understanding Occupancy

- **Definition**: Active warps ÷ Maximum possible warps per SM
- **Range**: 0% to 100%
- **Importance**: Higher occupancy can help hide memory and instruction latency

## Factors Limiting Occupancy

1. **Register Usage**
   - Each thread uses registers
   - Limited registers per SM (e.g., 65,536 on Ampere)
   - If each thread uses too many registers, fewer threads can run concurrently

2. **Shared Memory Usage**
   - Each block uses shared memory
   - Limited shared memory per SM (e.g., 48KB-164KB)
   - If each block uses too much shared memory, fewer blocks can run concurrently

3. **Block Size**
   - Must be a multiple of warp size (32) for best efficiency
   - Too small: Underutilizes resources
   - Too large: Limits the number of concurrent blocks

4. **Hardware Limits**
   - Maximum threads per SM
   - Maximum blocks per SM
   - Maximum warps per SM

## Optimizing Occupancy

1. **Reduce Register Usage**
   - Use \`__launch_bounds__\` to limit registers per thread
   - Break complex functions into simpler ones
   - Reuse variables when possible

2. **Optimize Shared Memory**
   - Only allocate what you need
   - Reuse shared memory for different purposes
   - Consider dynamic shared memory allocation

3. **Choose Appropriate Block Size**
   - Multiple of 32 (warp size)
   - Typically 128-512 threads per block
   - Balance between too small (underutilization) and too large (resource constraints)

4. **Use Occupancy Calculator**
   - CUDA provides tools to calculate theoretical occupancy
   - \`cudaOccupancyMaxPotentialBlockSize\` can suggest optimal block sizes

## Occupancy vs. Performance

- **Higher is not always better**: 100% occupancy is rarely needed
- **Diminishing returns**: Often 50-75% occupancy is sufficient
- **Trade-offs**: Sometimes lower occupancy with more resources per thread is better
- **Latency hiding**: The primary benefit of high occupancy is hiding latency

## Measuring Occupancy

- **Theoretical**: Calculate based on resource usage
- **Achieved**: Measure with CUDA profiling tools (Nsight, nvprof)
- **Dynamic**: Can vary during kernel execution
    `,
    code: `#include <stdio.h>

// Kernel with configurable register usage
__global__ void variableRegisterKernel(float* input, float* output, int n, int registerCount) {
    // Declare variables to force register usage
    float registers[32];  // Maximum of 32 registers we can control
    
    // Initialize registers
    for (int i = 0; i < registerCount && i < 32; i++) {
        registers[i] = i * 0.1f;
    }
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float result = input[idx];
        
        // Use the registers to prevent compiler optimization
        for (int i = 0; i < registerCount && i < 32; i++) {
            result += registers[i];
        }
        
        output[idx] = result;
    }
}

// Kernel with configurable shared memory usage
__global__ void variableSharedMemKernel(float* input, float* output, int n) {
    // Dynamic shared memory allocation
    extern __shared__ float sharedMem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && threadIdx.x < blockDim.x) {
        // Initialize shared memory
        sharedMem[threadIdx.x] = input[idx];
        __syncthreads();
        
        // Use shared memory
        float result = sharedMem[threadIdx.x];
        
        // Write result
        output[idx] = result;
    }
}

// Function to calculate theoretical occupancy
void calculateOccupancy(int blockSize, int registersPerThread, int sharedMemPerBlock) {
    int device;
    cudaDeviceProp prop;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate register-limited occupancy
    int registersPerBlock = registersPerThread * blockSize;
    int blocksPerSM_regs = prop.regsPerMultiprocessor / registersPerBlock;
    
    // Calculate shared memory-limited occupancy
    int blocksPerSM_shared = (int)(prop.sharedMemPerMultiprocessor / sharedMemPerBlock);
    
    // Calculate thread-limited occupancy
    int blocksPerSM_threads = prop.maxThreadsPerMultiProcessor / blockSize;
    
    // Take the minimum
    int blocksPerSM = min(min(blocksPerSM_regs, blocksPerSM_shared), blocksPerSM_threads);
    blocksPerSM = min(blocksPerSM, prop.maxBlocksPerMultiProcessor);
    
    // Calculate occupancy
    int warpsPerBlock = (blockSize + prop.warpSize - 1) / prop.warpSize;
    int activeWarps = blocksPerSM * warpsPerBlock;
    int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occupancy = (float)activeWarps / maxWarps;
    
    printf("Block Size: %d, Registers/Thread: %d, Shared Mem/Block: %d bytes\\n", 
           blockSize, registersPerThread, sharedMemPerBlock);
    printf("Blocks/SM: %d, Active Warps: %d, Max Warps: %d\\n", 
           blocksPerSM, activeWarps, maxWarps);
    printf("Theoretical Occupancy: %.2f%%\\n\\n", occupancy * 100);
}

// Function to find optimal block size using CUDA API
void findOptimalBlockSize() {
    int minGridSize;
    int blockSize;
    
    // Get optimal block size for variableRegisterKernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, 
        &blockSize, 
        (void*)variableRegisterKernel, 
        0,  // Dynamic shared memory size
        0); // No maximum block size limit
    
    printf("Optimal block size for variableRegisterKernel: %d\\n", blockSize);
    printf("Minimum grid size for full occupancy: %d\\n\\n", minGridSize);
}

int main() {
    // Data size
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    
    // Initialize input data
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 100;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Calculate theoretical occupancy for different configurations
    printf("Theoretical Occupancy Calculations:\\n\\n");
    
    // Different block sizes
    calculateOccupancy(128, 32, 4096);  // 128 threads, 32 registers/thread, 4KB shared mem
    calculateOccupancy(256, 32, 8192);  // 256 threads, 32 registers/thread, 8KB shared mem
    calculateOccupancy(512, 32, 16384); // 512 threads, 32 registers/thread, 16KB shared mem
    
    // Different register counts
    calculateOccupancy(256, 16, 8192);  // 256 threads, 16 registers/thread, 8KB shared mem
    calculateOccupancy(256, 32, 8192);  // 256 threads, 32 registers/thread, 8KB shared mem
    calculateOccupancy(256, 64, 8192);  // 256 threads, 64 registers/thread, 8KB shared mem
    
    // Find optimal block size using CUDA API
    findOptimalBlockSize();
    
    // Launch kernels with different configurations
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Test different register usage
    printf("Testing different register usage:\\n");
    for (int regs = 8; regs <= 32; regs += 8) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        variableRegisterKernel<<<gridSize, blockSize>>>(d_input, d_output, n, regs);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Register Count: %d, Execution Time: %.3f ms\\n", regs, milliseconds);
    }
    
    // Test different shared memory usage
    printf("\\nTesting different shared memory usage:\\n");
    for (int sharedMem = 1024; sharedMem <= 16384; sharedMem *= 2) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        variableSharedMemKernel<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Shared Memory: %d bytes, Execution Time: %.3f ms\\n", sharedMem, milliseconds);
    }
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}`,
  },
]

export default function CudaOptimizationPart1() {
  const [currentStep, setCurrentStep] = useState(0)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(cudaOptimizationPart1[currentStep].code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const nextStep = () => {
    if (currentStep < cudaOptimizationPart1.length - 1) {
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
              <CardTitle className="text-green-400">{cudaOptimizationPart1[currentStep].title}</CardTitle>
              <CardDescription>CUDA Optimization - Part 1</CardDescription>
            </div>
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {cudaOptimizationPart1.length}
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
                      __html: cudaOptimizationPart1[currentStep].content
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
                  {cudaOptimizationPart1[currentStep].code}
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
            disabled={currentStep === cudaOptimizationPart1.length - 1}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
