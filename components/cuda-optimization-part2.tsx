"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Copy, Check } from "lucide-react"

const cudaOptimizationPart2 = [
  {
    id: "global-memory-throughput",
    title: "Global Memory Throughput",
    content: `
# Global Memory Throughput

Global memory is the largest but slowest memory in the GPU. Optimizing global memory access patterns is crucial for high performance.

## Coalesced Memory Access

Coalesced memory access occurs when threads in a warp access contiguous memory locations, allowing the hardware to combine multiple memory transactions into fewer, larger transactions.

### Benefits of Coalesced Access:
- Maximizes memory bandwidth utilization
- Reduces number of memory transactions
- Improves overall performance

### Requirements for Coalesced Access:
1. **Alignment**: Memory addresses should be aligned to the transaction size
2. **Sequential Access**: Threads should access sequential elements
3. **Proper Stride**: Avoid non-unit strides between thread accesses

## Memory Access Patterns

### Good Patterns:
- **Sequential**: Thread i accesses element i
- **Aligned**: Starting address is a multiple of 128 bytes
- **Contiguous**: No gaps between accessed elements

### Poor Patterns:
- **Strided**: Thread i accesses element i*stride
- **Random**: Unpredictable access pattern
- **Misaligned**: Starting address is not properly aligned

## Memory Transaction Size

Memory transactions occur in units of:
- 32 bytes (L1 cache line size)
- 128 bytes (L2 cache line size)

When threads access data within the same cache line, they benefit from spatial locality.

## Strategies for Improving Global Memory Throughput

1. **Data Layout**: Structure data for coalesced access
2. **Array of Structures vs. Structure of Arrays**: Prefer SoA for better coalescing
3. **Memory Padding**: Ensure proper alignment
4. **Memory Access Reorganization**: Change algorithm to improve access patterns
5. **Use Shared Memory**: Load data into shared memory in a coalesced manner, then access freely
    `,
    code: `#include <stdio.h>

// Kernel with coalesced memory access
__global__ void coalescedAccess(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Coalesced access: adjacent threads access adjacent memory locations
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel with non-coalesced (strided) memory access
__global__ void stridedAccess(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Strided access: threads access memory with gaps between them
        int strided_idx = idx * stride;
        if (strided_idx < n) {
            output[strided_idx] = input[strided_idx] * 2.0f;
        }
    }
}

// Kernel demonstrating Array of Structures (AoS) access pattern
struct Particle {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
};

__global__ void aosPatterAccess(Particle* particles, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // AoS access: non-coalesced for individual components
        output[idx] = particles[idx].x * particles[idx].vx + 
                      particles[idx].y * particles[idx].vy + 
                      particles[idx].z * particles[idx].vz;
    }
}

// Kernel demonstrating Structure of Arrays (SoA) access pattern
struct ParticleSystem {
    float* x, *y, *z;    // Positions
    float* vx, *vy, *vz; // Velocities
};

__global__ void soaPatterAccess(ParticleSystem particles, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // SoA access: coalesced for individual components
        output[idx] = particles.x[idx] * particles.vx[idx] + 
                      particles.y[idx] * particles.vy[idx] + 
                      particles.z[idx] * particles.vz[idx];
    }
}

int main() {
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
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch coalesced access kernel and measure time
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    cudaEventRecord(start);
    coalescedAccess<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float coalescedTime = 0;
    cudaEventElapsedTime(&coalescedTime, start, stop);
    
    // Launch strided access kernels with different strides
    printf("Memory Access Pattern Performance:\\n");
    printf("Coalesced access: %.3f ms\\n", coalescedTime);
    
    int strides[] = {2, 4, 8, 16, 32};
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(start);
        stridedAccess<<<gridSize, blockSize>>>(d_input, d_output, n, strides[i]);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float stridedTime = 0;
        cudaEventElapsedTime(&stridedTime, start, stop);
        
        printf("Strided access (stride=%d): %.3f ms (%.2fx slower)\\n", 
               strides[i], stridedTime, stridedTime / coalescedTime);
    }
    
    // AoS vs SoA example
    int numParticles = 100000;
    
    // Allocate and initialize AoS data
    Particle* h_particles = (Particle*)malloc(numParticles * sizeof(Particle));
    for (int i = 0; i < numParticles; i++) {
        h_particles[i].x = i * 0.1f;
        h_particles[i].y = i * 0.2f;
        h_particles[i].z = i * 0.3f;
        h_particles[i].vx = i * 0.01f;
        h_particles[i].vy = i * 0.02f;
        h_particles[i].vz = i * 0.03f;
    }
    
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    
    // Allocate and initialize SoA data
    ParticleSystem h_particleSystem;
    h_particleSystem.x = (float*)malloc(numParticles * sizeof(float));
    h_particleSystem.y = (float*)malloc(numParticles * sizeof(float));
    h_particleSystem.z = (float*)malloc(numParticles * sizeof(float));
    h_particleSystem.vx = (float*)malloc(numParticles * sizeof(float));
    h_particleSystem.vy = (float*)malloc(numParticles * sizeof(float));
    h_particleSystem.vz = (float*)malloc(numParticles * sizeof(float));
    
    for (int i = 0; i < numParticles; i++) {
        h_particleSystem.x[i] = i * 0.1f;
        h_particleSystem.y[i] = i * 0.2f;
        h_particleSystem.z[i] = i * 0.3f;
        h_particleSystem.vx[i] = i * 0.01f;
        h_particleSystem.vy[i] = i * 0.02f;
        h_particleSystem.vz[i] = i * 0.03f;
    }
    
    ParticleSystem d_particleSystem;
    cudaMalloc(&d_particleSystem.x, numParticles * sizeof(float));
    cudaMalloc(&d_particleSystem.y, numParticles * sizeof(float));
    cudaMalloc(&d_particleSystem.z, numParticles * sizeof(float));
    cudaMalloc(&d_particleSystem.vx, numParticles * sizeof(float));
    cudaMalloc(&d_particleSystem.vy, numParticles * sizeof(float));
    cudaMalloc(&d_particleSystem.vz, numParticles * sizeof(float));
    
    cudaMemcpy(d_particleSystem.x, h_particleSystem.x, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleSystem.y, h_particleSystem.y, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleSystem.z, h_particleSystem.z, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleSystem.vx, h_particleSystem.vx, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleSystem.vy, h_particleSystem.vy, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleSystem.vz, h_particleSystem.vz, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compare AoS vs SoA performance
    gridSize = (numParticles + blockSize - 1) / blockSize;
    
    cudaEventRecord(start);
    aosPatterAccess<<<gridSize, blockSize>>>(d_particles, d_output, numParticles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float aosTime = 0;
    cudaEventElapsedTime(&aosTime, start, stop);
    
    cudaEventRecord(start);
    soaPatterAccess<<<gridSize, blockSize>>>(d_particleSystem, d_output, numParticles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float soaTime = 0;
    cudaEventElapsedTime(&soaTime, start, stop);
    
    printf("\\nData Layout Performance:\\n");
    printf("Array of Structures (AoS): %.3f ms\\n", aosTime);
    printf("Structure of Arrays (SoA): %.3f ms\\n", soaTime);
    printf("Speedup with SoA: %.2fx\\n", aosTime / soaTime);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_particles);
    cudaFree(d_particleSystem.x);
    cudaFree(d_particleSystem.y);
    cudaFree(d_particleSystem.z);
    cudaFree(d_particleSystem.vx);
    cudaFree(d_particleSystem.vy);
    cudaFree(d_particleSystem.vz);
    
    free(h_input);
    free(h_output);
    free(h_particles);
    free(h_particleSystem.x);
    free(h_particleSystem.y);
    free(h_particleSystem.z);
    free(h_particleSystem.vx);
    free(h_particleSystem.vy);
    free(h_particleSystem.vz);
    
    return 0;
}`,
  },
  {
    id: "aligned-access",
    title: "Importance of Aligned Memory Access",
    content: `
# Importance of Aligned Memory Access

Memory alignment refers to the way data is arranged and accessed in memory. Proper alignment is crucial for optimal performance on GPUs.

## Memory Alignment Basics

- **Definition**: Data is aligned when its memory address is a multiple of its size
- **Example**: A 4-byte float is aligned when its address is a multiple of 4
- **Cache Line**: Typically 32 bytes (L1) or 128 bytes (L2) on modern GPUs

## Why Alignment Matters

1. **Memory Transactions**:
   - GPU memory controllers operate on fixed-size transactions
   - Misaligned access may require multiple transactions
   - Each transaction has overhead

2. **Coalescing**:
   - Aligned access enables better coalescing
   - Coalesced access maximizes memory bandwidth utilization

3. **Cache Efficiency**:
   - Aligned data makes better use of cache lines
   - Reduces cache pollution and thrashing

## Alignment Requirements

- **Global Memory**: 128-byte alignment for optimal performance
- **Shared Memory**: 4-byte alignment to avoid bank conflicts
- **Constant Memory**: 4-byte alignment

## Techniques for Ensuring Alignment

1. **Allocation Functions**:
   - Use \`cudaMalloc\` (automatically aligns to largest scalar type)
   - For custom alignment, use \`cudaMallocPitch\` or \`cudaMalloc3D\`

2. **Data Padding**:
   - Add padding to structures to ensure alignment
   - Use \`__align__(n)\` directive

3. **Memory Access Patterns**:
   - Start memory accesses at aligned boundaries
   - Use vector types (float2, float4) for naturally aligned access

4. **2D Arrays**:
   - Use pitched memory for 2D arrays
   - Ensures each row starts at an aligned address

## Performance Impact

Misaligned access can reduce memory throughput by:
- 2-8x for older architectures
- 1.5-3x for newer architectures (which have better handling of misalignment)

The impact is most significant for memory-bound kernels.
    `,
    code: `#include <stdio.h>

// Structure with natural alignment
struct AlignedStruct {
    float x;  // 4 bytes
    float y;  // 4 bytes
    float z;  // 4 bytes
    float w;  // 4 bytes
};  // Total: 16 bytes, naturally aligned

// Structure with misalignment
struct MisalignedStruct {
    char flag;  // 1 byte
    float x;    // 4 bytes (potentially misaligned)
    float y;    // 4 bytes (potentially misaligned)
    float z;    // 4 bytes (potentially misaligned)
};  // Total: 13 bytes, with potential misalignment

// Structure with forced alignment
struct __align__(16) ForcedAlignedStruct {
    char flag;  // 1 byte
    float x;    // 4 bytes
    float y;    // 4 bytes
    float z;    // 4 bytes
    // 3 bytes of padding added automatically
};  // Total: 16 bytes, aligned to 16-byte boundary

// Kernel with aligned access
__global__ void alignedAccess(float4* input, float4* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Aligned access using float4 (16 bytes)
        float4 data = input[idx];
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;
        output[idx] = data;
    }
}

// Kernel with potentially misaligned access
__global__ void misalignedAccess(float* input, float* output, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n - offset) {
        // Misaligned access due to offset
        output[idx] = input[idx + offset] * 2.0f;
    }
}

// Kernel demonstrating pitched memory for 2D arrays
__global__ void pitched2DAccess(float* input, size_t pitch, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Properly aligned access using pitch
        float* row = (float*)((char*)input + y * pitch);
        output[y * width + x] = row[x] * 2.0f;
    }
}

int main() {
    int n = 1000000;
    
    // Allocate and initialize data for 1D array test
    float4* h_input4 = (float4*)malloc(n * sizeof(float4));
    float4* h_output4 = (float4*)malloc(n * sizeof(float4));
    
    for (int i = 0; i < n; i++) {
        h_input4[i].x = i * 0.1f;
        h_input4[i].y = i * 0.2f;
        h_input4[i].z = i * 0.3f;
        h_input4[i].w = i * 0.4f;
    }
    
    float4* d_input4;
    float4* d_output4;
    cudaMalloc(&d_input4, n * sizeof(float4));
    cudaMalloc(&d_output4, n * sizeof(float4));
    
    cudaMemcpy(d_input4, h_input4, n * sizeof(float4), cudaMemcpyHostToDevice);
    
    // Allocate and initialize data for misaligned access test
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_input[i] = i * 0.1f;
    }
    
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch aligned access kernel and measure time
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    cudaEventRecord(start);
    alignedAccess<<<gridSize, blockSize>>>(d_input4, d_output4, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float alignedTime = 0;
    cudaEventElapsedTime(&alignedTime, start, stop);
    
    // Launch misaligned access kernels with different offsets
    printf("Memory Alignment Performance:\\n");
    printf("Aligned access: %.3f ms\\n", alignedTime);
    
    int offsets[] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++) {
        cudaEventRecord(start);
        misalignedAccess<<<gridSize, blockSize>>>(d_input, d_output, n, offsets[i]);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float misalignedTime = 0;
        cudaEventElapsedTime(&misalignedTime, start, stop);
        
        printf("Misaligned access (offset=%d): %.3f ms (%.2fx slower)\\n", 
               offsets[i], misalignedTime, misalignedTime / alignedTime);
    }
    
    // 2D array with pitched memory
    int width = 1024;
    int height = 1024;
    
    float* h_input2D = (float*)malloc(width * height * sizeof(float));
    float* h_output2D = (float*)malloc(width * height * sizeof(float));
    
    for (int i = 0; i < width * height; i++) {
        h_input2D[i] = i % 100;
    }
    
    float* d_input2D;
    float* d_output2D;
    size_t pitch;
    
    // Allocate pitched memory
    cudaMallocPitch(&d_input2D, &pitch, width * sizeof(float), height);
    cudaMalloc(&d_output2D, width * height * sizeof(float));
    
    // Copy data to pitched memory
    cudaMemcpy2D(d_input2D, pitch, h_input2D, width * sizeof(float), 
                 width * sizeof(float), height, cudaMemcpyHostToDevice);
    
    // Launch kernel with pitched memory
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    cudaEventRecord(start);
    pitched2DAccess<<<grid, block>>>(d_input2D, pitch, d_output2D, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float pitchedTime = 0;
    cudaEventElapsedTime(&pitchedTime, start, stop);
    
    printf("\\n2D Array Access:\\n");
    printf("Pitched memory access: %.3f ms\\n", pitchedTime);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input4);
    cudaFree(d_output4);
    cudaFree(d_input2D);
    cudaFree(d_output2D);
    
    free(h_input);
    free(h_output);
    free(h_input4);
    free(h_output4);
    free(h_input2D);
    free(h_output2D);
    
    return 0;
}`,
  },
  {
    id: "shared-memory-access",
    title: "Shared Memory as a Fast Buffer",
    content: `
# Shared Memory as a Fast Buffer

Shared memory is a programmable on-chip memory that is much faster than global memory and can be used as a fast buffer for frequently accessed data.

## Shared Memory Characteristics

- **Speed**: ~100x faster than global memory
- **Size**: Limited (typically 48KB-64KB per SM)
- **Scope**: Accessible by all threads in a block
- **Lifetime**: Exists for the duration of the block execution

## Using Shared Memory as a Buffer

The typical pattern for using shared memory as a buffer:

1. **Load Phase**: Threads cooperatively load data from global memory into shared memory
2. **Synchronize**: Ensure all threads have completed loading with \`__syncthreads()\`
3. **Compute Phase**: Perform computations using data from shared memory
4. **Synchronize Again**: If needed, before writing results
5. **Store Phase**: Write results back to global memory

## Benefits of Shared Memory Buffering

1. **Reduced Global Memory Bandwidth Usage**:
   - Load data once, reuse multiple times
   - Especially beneficial for algorithms with data reuse

2. **Improved Memory Access Patterns**:
   - Load from global memory in a coalesced manner
   - Access from shared memory in any pattern without penalty

3. **Inter-Thread Communication**:
   - Enables threads to share intermediate results
   - Supports cooperative algorithms

## Common Use Cases

1. **Tiling**: Break large problems into smaller tiles that fit in shared memory
2. **Convolution/Stencil Operations**: Load neighborhood data once, reuse for multiple computations
3. **Reduction Operations**: Partial results stored in shared memory
4. **Matrix Operations**: Tile-based matrix multiplication, transposition
5. **Sorting Algorithms**: Shared memory for local sorting

## Performance Considerations

1. **Bank Conflicts**: Avoid multiple threads accessing the same bank
2. **Occupancy Impact**: Using more shared memory may reduce occupancy
3. **Synchronization Overhead**: \`__syncthreads()\` has a cost
4. **Shared Memory Size**: Limited resource, must be used efficiently
    `,
    code: `#include <stdio.h>

// Matrix multiplication without shared memory
__global__ void matrixMulNaive(float* A, float* B, float* C, int width) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix bounds
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        
        // Write result to C
        C[row * width + col] = sum;
    }
}

// Matrix multiplication with shared memory
__global__ void matrixMulShared(float* A, float* B, float* C, int width) {
    // Shared memory for tiles of A and B
    __shared__ float sharedA[32][32];
    __shared__ float sharedB[32][32];
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within tile
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Global row and column
    int globalRow = blockRow * blockDim.y + row;
    int globalCol = blockCol * blockDim.x + col;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (width + blockDim.x - 1) / blockDim.x; t++) {
        // Load tiles into shared memory
        if (globalRow < width && t * blockDim.x + col < width) {
            sharedA[row][col] = A[globalRow * width + t * blockDim.x + col];
        } else {
            sharedA[row][col] = 0.0f;
        }
        
        if (t * blockDim.y + row < width && globalCol < width) {
            sharedB[row][col] = B[(t * blockDim.y + row) * width + globalCol];
        } else {
            sharedB[row][col] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < blockDim.x; k++) {
            sum += sharedA[row][k] * sharedB[k][col];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to C
    if (globalRow < width && globalCol < width) {
        C[globalRow * width + globalCol] = sum;
    }
}

// Convolution kernel without shared memory
__global__ void convolutionNaive(float* input, float* output, float* filter, 
                                int width, int height, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        int filterRadius = filterWidth / 2;
        
        // Apply filter
        for (int fy = 0; fy < filterWidth; fy++) {
            for (int fx = 0; fx < filterWidth; fx++) {
                int inputRow = row + fy - filterRadius;
                int inputCol = col + fx - filterRadius;
                
                // Check bounds
                if (inputRow >= 0 && inputRow < height && 
                    inputCol >= 0 && inputCol < width) {
                    sum += input[inputRow * width + inputCol] * 
                           filter[fy * filterWidth + fx];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Convolution kernel with shared memory
__global__ void convolutionShared(float* input, float* output, float* filter, 
                                 int width, int height, int filterWidth) {
    // Shared memory for input tile and filter
    __shared__ float sharedInput[32+8][32+8];  // 32x32 tile with 4-pixel border
    __shared__ float sharedFilter[9][9];       // Maximum 9x9 filter
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int filterRadius = filterWidth / 2;
    
    // Load filter into shared memory (only done by some threads)
    if (tx < filterWidth && ty < filterWidth) {
        sharedFilter[ty][tx] = filter[ty * filterWidth + tx];
    }
    
    // Load input tile into shared memory (including halo region)
    int inputRow = row - filterRadius;
    int inputCol = col - filterRadius;
    
    // Each thread loads one element of the input tile
    if (inputRow >= 0 && inputRow < height && 
        inputCol >= 0 && inputCol < width) {
        sharedInput[ty][tx] = input[inputRow * width + inputCol];
    } else {
        sharedInput[ty][tx] = 0.0f;
    }
    
    // Additional loads for halo region (simplified)
    if (tx < 2*filterRadius && ty < blockDim.y) {
        int extraCol = col + blockDim.x;
        if (extraCol < width) {
            sharedInput[ty][tx + blockDim.x] = input[inputRow * width + extraCol];
        }
    }
    
    if (ty < 2*filterRadius && tx < blockDim.x) {
        int extraRow = row + blockDim.y;
        if (extraRow < height) {
            sharedInput[ty + blockDim.y][tx] = input[extraRow * width + inputCol];
        }
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Apply filter using shared memory
    if (col < width && row < height) {
        float sum = 0.0f;
        
        for (int fy = 0; fy < filterWidth; fy++) {
            for (int fx = 0; fx < filterWidth; fx++) {
                sum += sharedInput[ty + fy][tx + fx] * sharedFilter[fy][fx];
            }
        }
        
        output[row * width + col] = sum;
    }
}

int main() {
    // Matrix multiplication example
    int width = 1024;
    size_t bytes = width * width * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C1 = (float*)malloc(bytes);
    float* h_C2 = (float*)malloc(bytes);
    
    // Initialize matrices
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C1;
    float* d_C2;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C1, bytes);
    cudaMalloc(&d_C2, bytes);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch naive matrix multiplication
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (width + blockDim.y - 1) / blockDim.y);
    
    cudaEventRecord(start);
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C1, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveTime = 0;
    cudaEventElapsedTime(&naiveTime, start, stop);
    
    // Launch shared memory matrix multiplication
    cudaEventRecord(start);
    matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C2, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sharedTime = 0;
    cudaEventElapsedTime(&sharedTime, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results match
    bool resultsMatch = true;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C1[i] - h_C2[i]) > 1e-5) {
            resultsMatch = false;
            break;
        }
    }
    
    printf("Matrix Multiplication Performance:\\n");
    printf("Naive implementation: %.3f ms\\n", naiveTime);
    printf("Shared memory implementation: %.3f ms\\n", sharedTime);
    printf("Speedup: %.2fx\\n", naiveTime / sharedTime);
    printf("Results match: %s\\n\\n", resultsMatch ? "Yes" : "No");
    
    // Convolution example
    int imageWidth = 1024;
    int imageHeight = 1024;
    int filterWidth = 5;
    
    size_t imageBytes = imageWidth * imageHeight * sizeof(float);
    size_t filterBytes = filterWidth * filterWidth * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(imageBytes);
    float* h_output1 = (float*)malloc(imageBytes);
    float* h_output2 = (float*)malloc(imageBytes);
    float* h_filter = (float*)malloc(filterBytes);
    
    // Initialize data
    for (int i = 0; i < imageWidth * imageHeight; i++) {
        h_input[i] = i % 100;
    }
    
    // Simple Gaussian filter
    h_filter[0] = 1.0f; h_filter[1] = 2.0f; h_filter[2] = 3.0f; h_filter[3] = 2.0f; h_filter[4] = 1.0f;
    h_filter[5] = 2.0f; h_filter[6] = 3.0f; h_filter[7] = 4.0f; h_filter[8] = 3.0f; h_filter[9] = 2.0f;
    h_filter[10] = 3.0f; h_filter[11] = 4.0f; h_filter[12] = 5.0f; h_filter[13] = 4.0f; h_filter[14] = 3.0f;
    h_filter[15] = 2.0f; h_filter[16] = 3.0f; h_filter[17] = 4.0f; h_filter[18] = 3.0f; h_filter[19] = 2.0f;
    h_filter[20] = 1.0f; h_filter[21] = 2.0f; h_filter[22] = 3.0f; h_filter[23] = 2.0f; h_filter[24] = 1.0f;
    
    // Normalize filter
    float sum = 0.0f;
    for (int i = 0; i < filterWidth * filterWidth; i++) {
        sum += h_filter[i];
    }
    for (int i = 0; i < filterWidth * filterWidth; i++) {
        h_filter[i] /= sum;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output1;
    float* d_output2;
    float* d_filter;
    cudaMalloc(&d_input, imageBytes);
    cudaMalloc(&d_output1, imageBytes);
    cudaMalloc(&d_output2, imageBytes);
    cudaMalloc(&d_filter, filterBytes);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterBytes, cudaMemcpyHostToDevice);
    
    // Launch naive convolution
    dim3 convBlockDim(32, 32);
    dim3 convGridDim((imageWidth + convBlockDim.x - 1) / convBlockDim.x,
                     (imageHeight + convBlockDim.y - 1) / convBlockDim.y);
    
    cudaEventRecord(start);
    convolutionNaive<<<convGridDim, convBlockDim>>>(d_input, d_output1, d_filter, 
                                                  imageWidth, imageHeight, filterWidth);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveConvTime = 0;
    cudaEventElapsedTime(&naiveConvTime, start, stop);
    
    // Launch shared memory convolution
    cudaEventRecord(start);
    convolutionShared<<<convGridDim, convBlockDim>>>(d_input, d_output2, d_filter, 
                                                   imageWidth, imageHeight, filterWidth);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sharedConvTime = 0;
    cudaEventElapsedTime(&sharedConvTime, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_output1, d_output1, imageBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, imageBytes, cudaMemcpyDeviceToHost);
    
    printf("Convolution Performance:\\n");
    printf("Naive implementation: %.3f ms\\n", naiveConvTime);
    printf("Shared memory implementation: %.3f ms\\n", sharedConvTime);
    printf("Speedup: %.2fx\\n", naiveConvTime / sharedConvTime);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_filter);
    
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_filter);
    
    return 0;
}`,
  },
  {
    id: "bank-conflicts",
    title: "Bank Conflicts and Avoidance Strategies",
    content: `
# Bank Conflicts and Avoidance Strategies

Shared memory is divided into equally sized memory banks that can be accessed simultaneously by different threads. Bank conflicts occur when multiple threads in a warp access the same memory bank, causing serialization.

## Shared Memory Banks

- Modern GPUs typically have 32 banks (one for each thread in a warp)
- Each bank has a bandwidth of 32 bits (4 bytes) per clock cycle
- Successive 4-byte words are assigned to successive banks
- Bank index = (address / 4) % 32

## Types of Bank Conflicts

1. **No Conflict**: Each thread accesses a different bank
2. **2-Way Conflict**: 2 threads access the same bank
3. **N-Way Conflict**: N threads access the same bank
4. **Broadcast**: Multiple threads access the same address (no conflict)

## Common Causes of Bank Conflicts

1. **Strided Access**: When the stride is a multiple of the number of banks
2. **Random Access**: Unpredictable access patterns
3. **Row/Column Major Mismatch**: Accessing a row-major array column-wise
4. **Power-of-2 Strides**: Particularly problematic (e.g., stride of 8, 16, 32)

## Strategies to Avoid Bank Conflicts

1. **Padding**:
   - Add an extra element to each row of a 2D array
   - Changes the mapping of data to banks
   - Example: \`__shared__ float array[32][33];\` instead of \`[32][32]\`

2. **Shuffling**:
   - Reorganize data access patterns
   - Use different indexing schemes

3. **Warp-Aligned Access**:
   - Ensure threads in a warp access data with a stride that avoids conflicts
   - Use stride 32 for complete avoidance

4. **Use Warp Shuffle Instructions**:
   - For warp-level communication instead of shared memory
   - No bank conflicts by design

5. **Sequential Access**:
   - Prefer sequential access patterns when possible
   - Adjacent threads access adjacent memory locations

## Performance Impact

- Each conflict causes serialization of memory accesses
- N-way conflict results in N serialized transactions
- Impact increases with the degree of conflict
- Modern architectures have improved handling of conflicts
    `,
    code: `#include <stdio.h>

// Kernel with no bank conflicts
__global__ void noBankConflicts(float* input, float* output, int n) {
    __shared__ float sharedMem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory - no bank conflicts
    // Each thread accesses a different bank
    if (idx < n) {
        sharedMem[threadIdx.x] = input[idx];
    }
    
    __syncthreads();
    
    // Use shared memory - no bank conflicts
    // Each thread accesses a different bank
    if (idx < n) {
        output[idx] = sharedMem[threadIdx.x] * 2.0f;
    }
}

// Kernel with bank conflicts
__global__ void bankConflicts(float* input, float* output, int n, int stride) {
    __shared__ float sharedMem[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory - no bank conflicts
    if (idx < n) {
        sharedMem[threadIdx.x] = input[idx];
    }
    
    __syncthreads();
    
    // Use shared memory with strided access - potential bank conflicts
    if (idx < n) {
        int sharedIdx = (threadIdx.x * stride) % blockDim.x;
        output[idx] = sharedMem[sharedIdx] * 2.0f;
    }
}

// Kernel with bank conflicts in 2D array
__global__ void bankConflicts2D(float* input, float* output, int width, int height) {
    __shared__ float sharedMem[32][32];  // Potential bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load data into shared memory - no bank conflicts
    if (x < width && y < height) {
        sharedMem[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Access shared memory column-wise - bank conflicts
    if (x < width && y < height) {
        output[y * width + x] = sharedMem[threadIdx.x][threadIdx.y] * 2.0f;
    }
}

// Kernel with padded 2D array to avoid bank conflicts
__global__ void noBankConflicts2D(float* input, float* output, int width, int height) {
    __shared__ float sharedMem[32][33];  // Padded to avoid bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load data into shared memory - no bank conflicts
    if (x < width && y < height) {
        sharedMem[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Access shared memory column-wise - no bank conflicts due to padding
    if (x < width && y < height) {
        output[y * width + x] = sharedMem[threadIdx.x][threadIdx.y] * 2.0f;
    }
}

// Matrix transpose kernel with bank conflicts
__global__ void transposeWithConflicts(float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Load from input to shared memory
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Transpose block indices
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (x < height && y < width) {
        // Store from shared memory to output with transposed indices
        // This causes bank conflicts
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Matrix transpose kernel without bank conflicts
__global__ void transposeWithoutConflicts(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Padded to avoid bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Load from input to shared memory
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Transpose block indices
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (x < height && y < width) {
        // Store from shared memory to output with transposed indices
        // No bank conflicts due to padding
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
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
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel with no bank conflicts
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    cudaEventRecord(start);
    noBankConflicts<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float noConflictTime = 0;
    cudaEventElapsedTime(&noConflictTime, start, stop);
    
    // Launch kernels with different stride values (potential bank conflicts)
    printf("Bank Conflict Performance:\\n");
    printf("No bank conflicts: %.3f ms\\n", noConflictTime);
    
    int strides[] = {2, 4, 8, 16, 32};
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(start);
        bankConflicts<<<gridSize, blockSize>>>(d_input, d_output, n, strides[i]);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float conflictTime = 0;
        cudaEventElapsedTime(&conflictTime, start, stop);
        
        printf("Stride %d (potential conflicts): %.3f ms (%.2fx slower)\\n", 
               strides[i], conflictTime, conflictTime / noConflictTime);
    }
    
    // 2D array example
    int width = 1024;
    int height = 1024;
    size_t bytes2D = width * height * sizeof(float);
    
    // Allocate host memory
    float* h_input2D = (float*)malloc(bytes2D);
    float* h_output2D = (float*)malloc(bytes2D);
    
    // Initialize input data
    for (int i = 0; i < width * height; i++) {
        h_input2D[i] = i % 100;
    }
    
    // Allocate device memory
    float* d_input2D;
    float* d_output2D;
    cudaMalloc(&d_input2D, bytes2D);
    cudaMalloc(&d_output2D, bytes2D);
    
    // Copy input to device
    cudaMemcpy(d_input2D, h_input2D, bytes2D, cudaMemcpyHostToDevice);
    
    // Launch 2D kernels
    dim3 block2D(32, 32);
    dim3 grid2D((width + block2D.x - 1) / block2D.x, 
                (height + block2D.y - 1) / block2D.y);
    
    cudaEventRecord(start);
    bankConflicts2D<<<grid2D, block2D>>>(d_input2D, d_output2D, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float conflicts2DTime = 0;
    cudaEventElapsedTime(&conflicts2DTime, start, stop);
    
    cudaEventRecord(start);
    noBankConflicts2D<<<grid2D, block2D>>>(d_input2D, d_output2D, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float noConflicts2DTime = 0;
    cudaEventElapsedTime(&noConflicts2DTime, start, stop);
    
    printf("\\n2D Array Bank Conflicts:\\n");
    printf("With bank conflicts: %.3f ms\\n", conflicts2DTime);
    printf("Without bank conflicts (padded): %.3f ms\\n", noConflicts2DTime);
    printf("Speedup with padding: %.2fx\\n", conflicts2DTime / noConflicts2DTime);
    
    // Matrix transpose example
    cudaEventRecord(start);
    transposeWithConflicts<<<grid2D, block2D>>>(d_input2D, d_output2D, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float transposeConflictsTime = 0;
    cudaEventElapsedTime(&transposeConflictsTime, start, stop);
    
    cudaEventRecord(start);
    transposeWithoutConflicts<<<grid2D, block2D>>>(d_input2D, d_output2D, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float transposeNoConflictsTime = 0;
    cudaEventElapsedTime(&transposeNoConflictsTime, start, stop);
    
    printf("\\nMatrix Transpose Bank Conflicts:\\n");
    printf("With bank conflicts: %.3f ms\\n", transposeConflictsTime);
    printf("Without bank conflicts (padded): %.3f ms\\n", transposeNoConflictsTime);
    printf("Speedup with padding: %.2fx\\n", transposeConflictsTime / transposeNoConflictsTime);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input2D);
    cudaFree(d_output2D);
    
    free(h_input);
    free(h_output);
    free(h_input2D);
    free(h_output2D);
    
    return 0;
}`,
  },
]

export default function CudaOptimizationPart2() {
  const [currentStep, setCurrentStep] = useState(0)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(cudaOptimizationPart2[currentStep].code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const nextStep = () => {
    if (currentStep < cudaOptimizationPart2.length - 1) {
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
              <CardTitle className="text-green-400">{cudaOptimizationPart2[currentStep].title}</CardTitle>
              <CardDescription>CUDA Optimization - Part 2</CardDescription>
            </div>
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {cudaOptimizationPart2.length}
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
                      __html: cudaOptimizationPart2[currentStep].content
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
                  {cudaOptimizationPart2[currentStep].code}
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
            disabled={currentStep === cudaOptimizationPart2.length - 1}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
