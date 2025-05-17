"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Copy, Check } from "lucide-react"

const cudaSharedMemoryTutorial = [
  {
    id: "intro",
    title: "Introduction to CUDA Shared Memory",
    content: `
# Introduction to CUDA Shared Memory

Shared memory is a programmable on-chip memory that is much faster than global memory and is shared among all threads in a thread block.

## Key Characteristics

- **High Bandwidth**: Much faster than global memory (100x+ in some cases)
- **Low Latency**: Similar to L1 cache access times
- **Limited Size**: Typically 48KB-64KB per SM (shared among all active blocks on that SM)
- **Scope**: Only accessible within a thread block
- **Lifetime**: Exists only for the duration of the block execution

## Benefits of Using Shared Memory

1. **Reduced Global Memory Access**: Load data once into shared memory, then reuse it multiple times
2. **Data Sharing**: Enables threads within a block to share and cooperate on data
3. **Performance**: Can significantly improve kernel performance for algorithms with data reuse
4. **Bandwidth Optimization**: Reduces pressure on global memory bandwidth

## Declaring Shared Memory

Shared memory can be declared in two ways:

1. **Static allocation**:
   \`\`\`cuda
   __shared__ float shared_data[256];
   \`\`\`

2. **Dynamic allocation** (size determined at kernel launch):
   \`\`\`cuda
   extern __shared__ float shared_data[];
   \`\`\`
   
   Then specify the size when launching the kernel:
   \`\`\`cuda
   kernel<<<blocks, threads, sharedMemSize>>>(args);
   \`\`\`
    `,
    code: `#include <stdio.h>

// Kernel using shared memory
__global__ void sharedMemoryDemo(float* input, float* output, int n) {
    // Declare shared memory array
    __shared__ float sharedData[256];
    
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data from global to shared memory
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Use the shared data (just copy it to output in this example)
    if (idx < n) {
        output[idx] = sharedData[threadIdx.x];
    }
}

// Kernel using dynamically allocated shared memory
__global__ void dynamicSharedMemoryDemo(float* input, float* output, int n) {
    // Declare dynamic shared memory array
    extern __shared__ float sharedData[];
    
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data from global to shared memory
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Use the shared data (just copy it to output in this example)
    if (idx < n) {
        output[idx] = sharedData[threadIdx.x];
    }
}

int main() {
    // Array size
    int n = 1024;
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    
    // Initialize input array
    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel with static shared memory
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sharedMemoryDemo<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Launch kernel with dynamic shared memory
    int sharedMemSize = blockSize * sizeof(float);
    dynamicSharedMemoryDemo<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    
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
    id: "reduce-global-access",
    title: "Using Shared Memory to Reduce Global Memory Access",
    content: `
# Using Shared Memory to Reduce Global Memory Access

One of the primary uses of shared memory is to reduce global memory accesses by loading data once into shared memory and then reusing it multiple times.

## Matrix Multiplication Example

Matrix multiplication is a perfect example of how shared memory can improve performance:

1. **Without Shared Memory**:
   - Each thread reads multiple elements from global memory
   - For an NxN matrix multiplication, we perform N² reads from each input matrix

2. **With Shared Memory**:
   - Load blocks of the input matrices into shared memory
   - Each thread reads from shared memory instead of global memory
   - Reduces global memory accesses by a factor proportional to the block size

## Implementation Strategy

1. Divide input matrices into tiles that fit in shared memory
2. Each thread block loads one tile from each input matrix into shared memory
3. Threads compute partial results using the data in shared memory
4. Move to the next tile and repeat until all tiles are processed

## Performance Impact

Using shared memory for matrix multiplication can result in speedups of 10x or more compared to the naive implementation, especially for large matrices.
    `,
    code: `#include <stdio.h>

// Matrix multiplication kernel using shared memory
__global__ void matrixMulShared(float* A, float* B, float* C, int width) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread index
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Allocate shared memory for tiles
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    // Each thread computes one element of C
    float sum = 0.0f;
    
    // Loop over tiles of A and B
    for (int t = 0; t < (width + 15) / 16; ++t) {
        // Load tiles into shared memory
        if (blockRow * 16 + row < width && t * 16 + col < width)
            As[row][col] = A[(blockRow * 16 + row) * width + t * 16 + col];
        else
            As[row][col] = 0.0f;
            
        if (t * 16 + row < width && blockCol * 16 + col < width)
            Bs[row][col] = B[(t * 16 + row) * width + blockCol * 16 + col];
        else
            Bs[row][col] = 0.0f;
            
        // Synchronize to ensure all threads have loaded the tiles
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < 16; ++k) {
            sum += As[row][k] * Bs[k][col];
        }
        
        // Synchronize to ensure all threads are done with the tiles
        __syncthreads();
    }
    
    // Write result to global memory
    if (blockRow * 16 + row < width && blockCol * 16 + col < width)
        C[(blockRow * 16 + row) * width + blockCol * 16 + col] = sum;
}

// Matrix multiplication kernel without shared memory (for comparison)
__global__ void matrixMulNaive(float* A, float* B, float* C, int width) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix bounds
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        
        // Write result to C
        C[row * width + col] = sum;
    }
}

int main() {
    int width = 1024;  // Matrix dimensions
    size_t bytes = width * width * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (width + blockDim.y - 1) / blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch naive kernel and measure time
    cudaEventRecord(start);
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveTime = 0;
    cudaEventElapsedTime(&naiveTime, start, stop);
    
    // Launch shared memory kernel and measure time
    cudaEventRecord(start);
    matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sharedTime = 0;
    cudaEventElapsedTime(&sharedTime, start, stop);
    
    printf("Matrix multiplication timing:\\n");
    printf("Naive implementation: %.3f ms\\n", naiveTime);
    printf("Shared memory implementation: %.3f ms\\n", sharedTime);
    printf("Speedup: %.2fx\\n", naiveTime / sharedTime);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}`,
  },
  {
    id: "stencil-sync",
    title: "Stencil Operations and Synchronization",
    content: `
# Stencil Operations and Synchronization

Stencil operations are common in scientific computing and image processing. They involve updating each element based on its neighbors (e.g., convolution, blur filters, differential equations).

## Stencil Operations

In a stencil operation:
- Each output element depends on multiple input elements
- There's a pattern (stencil) that defines which neighbors are used
- Examples: 3x3 blur filter, 5-point stencil for PDEs, convolution kernels

## Need for Synchronization

When using shared memory for stencil operations:

1. Threads load data into shared memory
2. **Synchronization is required** to ensure all data is loaded before computation
3. Threads compute results using the shared memory data
4. Write results back to global memory

## The __syncthreads() Function

\`__syncthreads()\` is a barrier synchronization function that ensures all threads in a block have reached the same point before continuing.

Key points:
- Affects only threads within the same block
- All threads must execute the same \`__syncthreads()\` statement
- Cannot be used to synchronize across different blocks
- Should not be placed in divergent code paths (if/else where some threads execute it and others don't)

## Common Pattern for Stencil Operations

1. Load input data into shared memory (including halo/ghost cells)
2. Call \`__syncthreads()\` to ensure all data is loaded
3. Perform stencil computation using shared memory
4. Call \`__syncthreads()\` again if needed
5. Write results to global memory
    `,
    code: `#include <stdio.h>

// 1D stencil kernel using shared memory
__global__ void stencil1D(float* in, float* out, int width) {
    // Shared memory for input elements including halo regions
    __shared__ float temp[256 + 2];  // 256 elements + 2 halo elements
    
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + 1;  // Local index in shared memory (offset by 1 for halo)
    
    // Load regular cells
    if (gindex < width) {
        temp[lindex] = in[gindex];
    }
    
    // Load halo cells
    if (threadIdx.x == 0 && gindex > 0) {
        // Left halo
        temp[0] = in[gindex - 1];
    }
    
    if (threadIdx.x == blockDim.x - 1 && gindex < width - 1) {
        // Right halo
        temp[blockDim.x + 1] = in[gindex + 1];
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Apply 3-point stencil
    if (gindex < width) {
        out[gindex] = 0.25f * temp[lindex - 1] + 0.5f * temp[lindex] + 0.25f * temp[lindex + 1];
    }
}

// 2D stencil kernel (5-point) using shared memory
__global__ void stencil2D(float* in, float* out, int width, int height) {
    // Shared memory for input elements including halo regions
    __shared__ float temp[18][18];  // 16x16 block + 2x2 halo
    
    // Global indices
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local indices (with offset for halo)
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
    
    // Load regular cells
    if (gx < width && gy < height) {
        temp[ly][lx] = in[gy * width + gx];
    }
    
    // Load halo cells (simplified - doesn't handle all edge cases)
    
    // Top edge
    if (threadIdx.y == 0 && gy > 0) {
        temp[0][lx] = in[(gy - 1) * width + gx];
    }
    
    // Bottom edge
    if (threadIdx.y == blockDim.y - 1 && gy < height - 1) {
        temp[ly + 1][lx] = in[(gy + 1) * width + gx];
    }
    
    // Left edge
    if (threadIdx.x == 0 && gx > 0) {
        temp[ly][0] = in[gy * width + (gx - 1)];
    }
    
    // Right edge
    if (threadIdx.x == blockDim.x - 1 && gx < width - 1) {
        temp[ly][lx + 1] = in[gy * width + (gx + 1)];
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Apply 5-point stencil
    if (gx < width && gy < height) {
        out[gy * width + gx] = 0.2f * (temp[ly][lx] +     // Center
                                       temp[ly-1][lx] +   // Top
                                       temp[ly+1][lx] +   // Bottom
                                       temp[ly][lx-1] +   // Left
                                       temp[ly][lx+1]);   // Right
    }
}

int main() {
    // 1D stencil example
    int width = 1024;
    size_t bytes = width * sizeof(float);
    
    // Allocate host memory
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    
    // Initialize input array
    for (int i = 0; i < width; i++) {
        h_in[i] = i;
    }
    
    // Allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    // Launch 1D stencil kernel
    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;
    stencil1D<<<gridSize, blockSize>>>(d_in, d_out, width);
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    // Verify first few results
    printf("1D Stencil Results (first 5 elements):\\n");
    for (int i = 0; i < 5; i++) {
        printf("out[%d] = %.2f\\n", i, h_out[i]);
    }
    
    // 2D stencil example
    int height = 1024;
    bytes = width * height * sizeof(float);
    
    // Reallocate host memory
    free(h_in);
    free(h_out);
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    
    // Initialize input array
    for (int i = 0; i < width * height; i++) {
        h_in[i] = i % 100;
    }
    
    // Reallocate device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    // Launch 2D stencil kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    stencil2D<<<grid, block>>>(d_in, d_out, width, height);
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    
    return 0;
}`,
  },
  {
    id: "race-conditions",
    title: "Avoiding Race Conditions and Bank Conflicts",
    content: `
# Avoiding Race Conditions and Bank Conflicts

## Race Conditions

A race condition occurs when multiple threads access and modify the same memory location without proper synchronization.

### Common Causes of Race Conditions:
1. Multiple threads writing to the same shared memory location
2. One thread reading while another is writing
3. Missing \`__syncthreads()\` calls

### Avoiding Race Conditions:
1. Use \`__syncthreads()\` to synchronize threads
2. Use atomic operations for shared updates
3. Ensure each thread writes to a unique location
4. Use proper memory fences when needed

## Bank Conflicts

Shared memory is divided into equally sized memory banks that can be accessed simultaneously by different threads.

### What are Bank Conflicts?
- Bank conflicts occur when multiple threads in a warp access different addresses in the same memory bank
- This causes serialization of memory accesses, reducing performance
- On modern GPUs, shared memory typically has 32 banks (one for each thread in a warp)

### Types of Bank Conflicts:
1. **2-way conflict**: 2 threads access the same bank
2. **N-way conflict**: N threads access the same bank

### Avoiding Bank Conflicts:
1. **Padding**: Add extra elements to change the mapping of data to banks
2. **Shuffling**: Reorganize data access patterns
3. **Use warp-aligned access patterns**: Each thread accesses elements with stride 32
4. **Use shuffle instructions**: For warp-level communication instead of shared memory

## Example: Padding to Avoid Bank Conflicts

For a 2D array in shared memory, adding padding can help avoid bank conflicts:

\`\`\`cuda
// With bank conflicts (for 32 banks)
__shared__ float data[32][32];

// Without bank conflicts (padding by 1)
__shared__ float data[32][33];  // 33 columns instead of 32
\`\`\`

The extra column ensures that rows are not aligned with the same banks.
    `,
    code: `#include <stdio.h>

// Kernel demonstrating bank conflicts
__global__ void bankConflictsDemo(float* input, float* output, int n) {
    // Shared memory with potential bank conflicts
    __shared__ float sharedData[32][32];
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ float sharedDataPadded[32][33];  // 33 columns instead of 32
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = row * 32 + col;
    
    // Load data into shared memory (both versions)
    if (idx < n) {
        sharedData[row][col] = input[idx];
        sharedDataPadded[row][col] = input[idx];
    }
    
    __syncthreads();
    
    // Example 1: Column-wise access with bank conflicts
    // Each thread in a warp accesses the same bank
    float sum1 = 0.0f;
    if (col < 32) {
        for (int i = 0; i < 32; i++) {
            sum1 += sharedData[i][col];  // Bank conflicts occur here
        }
    }
    
    // Example 2: Column-wise access without bank conflicts using padding
    float sum2 = 0.0f;
    if (col < 32) {
        for (int i = 0; i < 32; i++) {
            sum2 += sharedDataPadded[i][col];  // No bank conflicts
        }
    }
    
    // Store results
    if (idx < n) {
        output[idx] = sum1 + sum2;
    }
}

// Kernel demonstrating race conditions and how to avoid them
__global__ void raceConditionDemo(int* counter) {
    // This will cause a race condition
    // Multiple threads increment the same memory location
    (*counter)++;
    
    // Proper way to increment using atomic operations
    atomicAdd(counter + 1, 1);
}

int main() {
    // Bank conflicts example
    int n = 1024;
    size_t bytes = n * sizeof(float);
    
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }
    
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(32, 32);
    dim3 grid(1, 1);
    bankConflictsDemo<<<grid, block>>>(d_input, d_output, n);
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Race condition example
    int h_counter[2] = {0, 0};
    int* d_counter;
    cudaMalloc(&d_counter, 2 * sizeof(int));
    cudaMemcpy(d_counter, h_counter, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple threads that will cause race conditions
    raceConditionDemo<<<1, 256>>>(d_counter);
    
    cudaMemcpy(h_counter, d_counter, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Race condition demonstration:\\n");
    printf("Counter with race condition: %d\\n", h_counter[0]);
    printf("Counter with atomic operations: %d\\n", h_counter[1]);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_counter);
    free(h_input);
    free(h_output);
    
    return 0;
}`,
  },
  {
    id: "padding-techniques",
    title: "Padding Techniques for Shared Memory",
    content: `
# Padding Techniques for Shared Memory

Padding is a technique used to avoid bank conflicts in shared memory by adding extra elements to change the memory access pattern.

## Why Padding Works

- Shared memory is divided into banks (typically 32 banks on modern GPUs)
- Each bank can service one memory request per cycle
- If multiple threads access the same bank, the accesses are serialized
- Padding changes the mapping of data to banks, reducing conflicts

## Common Padding Techniques

### 1. Array Padding

Add extra elements to the end of each row in a 2D array:

\`\`\`cuda
// Without padding (potential bank conflicts)
__shared__ float data[32][32];

// With padding (avoids bank conflicts)
__shared__ float data[32][33];  // One extra column
\`\`\`

### 2. Stride Padding

Change the access pattern by using a stride that avoids conflicts:

\`\`\`cuda
// Access with potential bank conflicts
float value = sharedData[threadIdx.x];

// Access with stride to avoid conflicts
float value = sharedData[threadIdx.x * (BANK_SIZE + 1) % ARRAY_SIZE];
\`\`\`

### 3. Dynamic Padding

Calculate the optimal padding at runtime based on the problem size:

\`\`\`cuda
int paddedWidth = width + (width % 32 == 0 ? 1 : 0);
\`\`\`

## Performance Impact

- Padding increases shared memory usage slightly
- But it can significantly improve performance by avoiding bank conflicts
- The performance gain usually outweighs the extra memory usage
- Modern CUDA architectures (compute capability 7.0+) have improved bank conflict handling

## When to Use Padding

- When your algorithm shows poor performance due to shared memory access patterns
- When profiling indicates bank conflicts as a bottleneck
- For matrix transposition and similar operations with cross-thread access patterns
- When working with power-of-2 sized data structures (which often align with bank boundaries)
    `,
    code: `#include <stdio.h>
#include <cuda_runtime.h>

// Matrix transpose kernel without padding (has bank conflicts)
__global__ void transposeNaive(float* input, float* output, int width, int height) {
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
        // This causes bank conflicts because threads in the same warp
        // access the same bank
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Matrix transpose kernel with padding to avoid bank conflicts
__global__ void transposePadded(float* input, float* output, int width, int height) {
    // Add padding to avoid bank conflicts
    __shared__ float tile[32][33];  // 33 columns instead of 32
    
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
        // The padding prevents bank conflicts
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    size_t bytes = width * height * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float* h_output1 = (float*)malloc(bytes);
    float* h_output2 = (float*)malloc(bytes);
    
    // Initialize input matrix
    for (int i = 0; i < width * height; i++) {
        h_input[i] = i;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output1;
    float* d_output2;
    checkCudaError(cudaMalloc(&d_input, bytes), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output1, bytes), "cudaMalloc d_output1");
    checkCudaError(cudaMalloc(&d_output2, bytes), "cudaMalloc d_output2");
    
    // Copy input to device
    checkCudaError(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    
    // Define grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch naive transpose kernel and measure time
    cudaEventRecord(start);
    transposeNaive<<<gridDim, blockDim>>>(d_input, d_output1, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveTime = 0;
    cudaEventElapsedTime(&naiveTime, start, stop);
    
    // Launch padded transpose kernel and measure time
    cudaEventRecord(start);
    transposePadded<<<gridDim, blockDim>>>(d_input, d_output2, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float paddedTime = 0;
    cudaEventElapsedTime(&paddedTime, start, stop);
    
    // Copy results back to host
    checkCudaError(cudaMemcpy(h_output1, d_output1, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H 1");
    checkCudaError(cudaMemcpy(h_output2, d_output2, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H 2");
    
    // Verify results match
    bool resultsMatch = true;
    for (int i = 0; i < width * height; i++) {
        if (h_output1[i] != h_output2[i]) {
            resultsMatch = false;
            printf("Mismatch at index %d: %f vs %f\\n", i, h_output1[i], h_output2[i]);
            break;
        }
    }
    
    printf("Matrix transpose timing:\\n");
    printf("Naive implementation: %.3f ms\\n", naiveTime);
    printf("Padded implementation: %.3f ms\\n", paddedTime);
    printf("Speedup: %.2fx\\n", naiveTime / paddedTime);
    printf("Results match: %s\\n", resultsMatch ? "Yes" : "No");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    free(h_input);
    free(h_output1);
    free(h_output2);
    
    return 0;
}`,
  },
]

export default function CudaSharedMemoryTutorial() {
  const [currentStep, setCurrentStep] = useState(0)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(cudaSharedMemoryTutorial[currentStep].code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const nextStep = () => {
    if (currentStep < cudaSharedMemoryTutorial.length - 1) {
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
              <CardTitle className="text-green-400">{cudaSharedMemoryTutorial[currentStep].title}</CardTitle>
              <CardDescription>CUDA Shared Memory</CardDescription>
            </div>
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {cudaSharedMemoryTutorial.length}
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
                      __html: cudaSharedMemoryTutorial[currentStep].content
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
                <pre className="bg-white text-black p-4 rounded-lg overflow-x-auto">
                  {cudaSharedMemoryTutorial[currentStep].code}
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
            disabled={currentStep === cudaSharedMemoryTutorial.length - 1}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
