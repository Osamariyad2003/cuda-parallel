"use client"

import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Play, Copy, Check, RefreshCw } from "lucide-react"
import { compileCudaCode } from "@/services/cuda-compiler"

interface CodeExample {
  id: string;
  name: string;
  code: string;
}

// Predefined CUDA code examples
const codeExamples: CodeExample[] = [
  {
    id: "vector-add",
    name: "Vector Addition",
    code: `#include <stdio.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Vector size
    int n = 10;
    size_t bytes = n * sizeof(float);
    
    // Host vectors
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize vectors on host
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate memory on the device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Vector Addition Results:\\n");
    for (int i = 0; i < n; i++) {
        printf("h_c[%d] = %.1f\\n", i, h_c[i]);
    }
    
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
  },
  {
    id: "matrix-mul",
    name: "Matrix Multiplication",
    code: `#include <stdio.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *a, float *b, float *c, int width) {
    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    // Matrix dimensions
    int width = 4;
    size_t bytes = width * width * sizeof(float);
    
    // Host matrices
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize matrices on host
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            h_a[i * width + j] = i + j;
            h_b[i * width + j] = i - j;
        }
    }
    
    // Device matrices
    float *d_a, *d_b, *d_c;
    
    // Allocate memory on the device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(2, 2);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (width + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Matrix Multiplication Results:\\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f\\t", h_c[i * width + j]);
        }
        printf("\\n");
    }
    
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
  },
  {
    id: "shared-memory",
    name: "Shared Memory Example",
    code: `#include <stdio.h>

// CUDA kernel using shared memory
__global__ void sharedMemoryExample(float *input, float *output, int n) {
    // Declare shared memory array
    __shared__ float sharedData[256];
    
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Process data in shared memory
    if (idx < n && threadIdx.x < blockDim.x - 1) {
        // Simple stencil operation: average with right neighbor
        output[idx] = (sharedData[threadIdx.x] + sharedData[threadIdx.x + 1]) / 2.0f;
    } else if (idx < n) {
        output[idx] = sharedData[threadIdx.x];
    }
}

int main() {
    // Array size
    int n = 10;
    size_t bytes = n * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input array
    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }
    
    // Device arrays
    float *d_input, *d_output;
    
    // Allocate memory on the device
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sharedMemoryExample<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Shared Memory Example Results:\\n");
    printf("Input\\tOutput\\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f\\t%.1f\\n", h_input[i], h_output[i]);
    }
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    return 0;
}`,
  },
  {
    id: "atomic-operations",
    name: "Atomic Operations",
    code: `#include <stdio.h>

// CUDA kernel with atomic operations
__global__ void atomicExample(int *counter, int n) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Atomically increment the counter
        atomicAdd(counter, 1);
        
        // Atomically add the thread index to the second counter
        atomicAdd(counter + 1, idx);
        
        // Atomically compute the maximum value
        atomicMax(counter + 2, idx);
    }
}

int main() {
    // Number of threads
    int n = 1000;
    
    // Host counters
    int h_counter[3] = {0, 0, 0};
    
    // Device counter
    int *d_counter;
    
    // Allocate memory on the device
    cudaMalloc(&d_counter, 3 * sizeof(int));
    
    // Copy initial counter value to device
    cudaMemcpy(d_counter, h_counter, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    atomicExample<<<gridSize, blockSize>>>(d_counter, n);
    
    // Copy result back to host
    cudaMemcpy(h_counter, d_counter, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Atomic Operations Results:\\n");
    printf("Counter (atomicAdd with 1): %d (expected %d)\\n", h_counter[0], n);
    printf("Counter (atomicAdd with idx): %d\\n", h_counter[1]);
    printf("Counter (atomicMax): %d (expected %d)\\n", h_counter[2], n-1);
    
    // Free device memory
    cudaFree(d_counter);
    
    return 0;
}`,
  },
]

const CudaCodeSimulator: React.FC = () => {
  const [selectedExample, setSelectedExample] = useState<CodeExample>(codeExamples[0])
  const [output, setOutput] = useState<string>("")
  const [isRunning, setIsRunning] = useState<boolean>(false)
  const [copied, setCopied] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<string>("code")

  const handleRun = async () => {
    setIsRunning(true)
    setOutput("")
    setActiveTab("output")

    try {
      // Compile and run the CUDA code
      const result = await compileCudaCode(selectedExample.code)
      
      if (result.success) {
        setOutput(result.output)
      } else {
        setOutput(`Compilation/Execution Error:\n${result.error}`)
      }
    } catch (error) {
      setOutput("Error executing code. Please make sure CUDA is properly installed.")
    }

    setIsRunning(false)
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedExample.code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Card className="bg-gray-800 border-gray-700 shadow-lg">
      <CardHeader className="bg-gray-700 border-b border-gray-600">
        <div className="flex justify-between items-center">
          <CardTitle className="text-green-400">CUDA Code Simulator</CardTitle>
          <div className="flex space-x-2">
            <select
              className="bg-gray-800 text-gray-200 text-sm rounded border border-gray-600 px-2 py-1"
              value={selectedExample.id}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                const selected = codeExamples.find((ex) => ex.id === e.target.value)
                if (selected) {
                  setSelectedExample(selected)
                  setOutput("")
                }
              }}
            >
              {codeExamples.map((example) => (
                <option key={example.id} value={example.id}>
                  {example.name}
                </option>
              ))}
            </select>
            <Button
              variant="outline"
              size="sm"
              className="bg-gray-800 border-gray-600 text-gray-200 hover:bg-gray-700"
              onClick={handleCopy}
            >
              {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="w-full bg-gray-700 rounded-none">
            <TabsTrigger value="code" className="flex-1">
              Code
            </TabsTrigger>
            <TabsTrigger value="output" className="flex-1">
              Output
            </TabsTrigger>
          </TabsList>
          <TabsContent value="code" className="m-0">
            <div className="bg-gray-900 p-4 overflow-x-auto max-h-96 overflow-y-auto">
              <pre className="text-gray-300 text-sm">
                <code>{selectedExample.code}</code>
              </pre>
            </div>
          </TabsContent>
          <TabsContent value="output" className="m-0">
            <div className="bg-black p-4 font-mono text-sm text-green-400 max-h-96 overflow-y-auto">
              {output ? <pre>{output}</pre> : <div className="text-gray-500 italic">Run the code to see output</div>}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="bg-gray-700 border-t border-gray-600 p-3">
        <Button onClick={handleRun} disabled={isRunning} className="bg-green-500 hover:bg-green-600 text-black">
          {isRunning ? (
            <>
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Compiling & Running...
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Run Code
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  )
}

export default CudaCodeSimulator;
