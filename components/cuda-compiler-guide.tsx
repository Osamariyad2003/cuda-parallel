"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Copy, Check } from "lucide-react"

const cudaCompilerSteps = [
  {
    id: "intro",
    title: "CUDA Compilation Workflow",
    content: `
# CUDA Compilation Workflow

Compiling CUDA code involves using NVIDIA's CUDA compiler driver, \`nvcc\`. This tool handles the separation of host code (CPU) and device code (GPU), compiling each with the appropriate compiler.

## Basic Compilation Process

1. **Write your CUDA code** with \`.cu\` file extension
2. **Compile with nvcc** to generate an executable
3. **Run the executable** on a system with CUDA-capable GPU

## NVCC Compiler Driver

\`nvcc\` is the NVIDIA CUDA Compiler driver that separates device code from host code:

- **Device code**: Compiled by NVIDIA's proprietary compiler
- **Host code**: Compiled by the host compiler (e.g., gcc, clang, MSVC)

## Compilation Flags

Common \`nvcc\` flags include:

- **-o**: Specify output file name
- **-arch=sm_XX**: Specify target GPU architecture (e.g., sm_70 for Volta)
- **-G**: Generate debug information
- **-O3**: Enable high-level optimizations
- **-Xptxas**: Pass options to the PTX assembler
- **-I**: Add include directory
- **-l**: Link with a library

## Example Compilation Command

\`\`\`bash
nvcc -arch=sm_70 -O3 my_cuda_program.cu -o my_cuda_program
\`\`\`

This command compiles \`my_cuda_program.cu\` with optimizations for the Volta architecture and creates an executable named \`my_cuda_program\`.
    `,
    code: `# Basic CUDA compilation
nvcc -o vector_add vector_add.cu

# Compilation with architecture specification
nvcc -arch=sm_70 -o vector_add vector_add.cu

# Compilation with optimization flags
nvcc -O3 -o vector_add vector_add.cu

# Compilation with debugging information
nvcc -G -o vector_add vector_add.cu

# Compilation with multiple source files
nvcc -o my_program main.cu kernel.cu utils.cu

# Compilation with include directories and libraries
nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -o matrix_mul matrix_mul.cu`,
  },
  {
    id: "makefile",
    title: "Using Makefiles for CUDA Projects",
    content: `
# Using Makefiles for CUDA Projects

For larger CUDA projects, using a Makefile can simplify the build process and manage dependencies.

## Benefits of Using Makefiles

- **Automation**: Automate the build process
- **Dependency Management**: Only rebuild what's necessary
- **Consistency**: Ensure consistent build flags
- **Portability**: Work across different systems

## Basic Makefile Structure for CUDA

A basic Makefile for a CUDA project typically includes:

1. **Variables**: Define compiler, flags, and file lists
2. **Rules**: Specify how to build targets
3. **Dependencies**: Track file dependencies

## Example Makefile

\`\`\`makefile
# CUDA Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O3 -arch=sm_70

# Include and library directories
INCLUDES := -I/usr/local/cuda/include
LIBRARIES := -L/usr/local/cuda/lib64 -lcudart

# Source files and output
SOURCES := main.cu kernel.cu
TARGET := cuda_program

# Build rule
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LIBRARIES)

# Clean rule
clean:
	rm -f $(TARGET)
\`\`\`

## Using the Makefile

To build the project:
\`\`\`bash
make
\`\`\`

To clean the build:
\`\`\`bash
make clean
\`\`\`
    `,
    code: `# Example Makefile for a CUDA project

# CUDA Compiler and flags
NVCC := nvcc
NVCC_FLAGS := -O3 -arch=sm_70

# Directories
CUDA_ROOT := /usr/local/cuda
INCLUDES := -I$(CUDA_ROOT)/include -I./include
LIBRARIES := -L$(CUDA_ROOT)/lib64 -lcudart -lcublas

# Source files
CU_SOURCES := $(wildcard src/*.cu)
CU_OBJECTS := $(CU_SOURCES:.cu=.o)

# Output executable
TARGET := cuda_application

# Default target
all: $(TARGET)

# Link rule
$(TARGET): $(CU_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(LIBRARIES) $^ -o $@

# Compilation rule
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(CU_OBJECTS) $(TARGET)

# Run rule
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: NVCC_FLAGS += -G -g
debug: clean all`,
  },
  {
    id: "compilation-errors",
    title: "Common CUDA Compilation Errors",
    content: `
# Common CUDA Compilation Errors

When compiling CUDA code, you may encounter various errors. Here are some common issues and how to resolve them.

## 1. Missing CUDA Toolkit

**Error**: \`nvcc: command not found\`

**Solution**: 
- Install the CUDA Toolkit
- Add CUDA bin directory to your PATH:
  \`\`\`bash
  export PATH=/usr/local/cuda/bin:$PATH
  \`\`\`

## 2. Architecture Mismatch

**Error**: \`sm_XX is not a defined architecture\`

**Solution**:
- Use an architecture supported by your CUDA version
- Check your GPU's compute capability and use the appropriate -arch flag

## 3. Host/Device Function Errors

**Error**: \`calling a __host__ function from a __device__ function\`

**Solution**:
- Add \`__host__ __device__\` to functions called from both host and device code
- Separate host and device functionality

## 4. Memory Allocation Failures

**Error**: \`cudaMalloc returned error code X\`

**Solution**:
- Check for sufficient GPU memory
- Add error checking after CUDA API calls
- Use \`cudaGetLastError()\` to get detailed error information

## 5. Undefined CUDA Symbols

**Error**: \`undefined reference to 'cudaXXX'\`

**Solution**:
- Link against the CUDA runtime library: \`-lcudart\`
- Check for typos in CUDA API function names

## 6. Kernel Launch Failures

**Error**: \`an illegal memory access was encountered\` or \`launch failed\`

**Solution**:
- Check kernel launch parameters (grid and block dimensions)
- Verify memory allocations and accesses
- Use \`cuda-memcheck\` to debug memory issues

## 7. Debugging Tips

- Use \`-G\` flag to enable debugging information
- Use \`cuda-gdb\` for debugging CUDA applications
- Use \`nvprof\` or \`nsight\` for performance analysis
    `,
    code: `// Example of error checking in CUDA code

#include <stdio.h>

// Helper macro for error checking
#define cudaCheckError() {                                          \\
    cudaError_t e = cudaGetLastError();                             \\
    if (e != cudaSuccess) {                                         \\
        printf("CUDA error %s:%d: %s\\n", __FILE__, __LINE__,       \\
               cudaGetErrorString(e));                              \\
        exit(EXIT_FAILURE);                                         \\
    }                                                               \\
}

int main() {
    // Allocate device memory
    float* d_data;
    cudaMalloc((void**)&d_data, 1000 * sizeof(float));
    cudaCheckError();  // Check for allocation errors
    
    // Launch kernel with error checking
    dim3 blockSize(256);
    dim3 gridSize((1000 + blockSize.x - 1) / blockSize.x);
    myKernel<<<gridSize, blockSize>>>(d_data, 1000);
    cudaCheckError();  // Check for launch errors
    
    // Synchronize and check for runtime errors
    cudaDeviceSynchronize();
    cudaCheckError();  // Check for runtime errors
    
    // Clean up
    cudaFree(d_data);
    cudaCheckError();  // Check for deallocation errors
    
    return 0;
}`,
  },
  {
    id: "runtime-execution",
    title: "Running and Profiling CUDA Applications",
    content: `
# Running and Profiling CUDA Applications

After successfully compiling your CUDA application, the next steps are running it efficiently and analyzing its performance.

## Running CUDA Applications

To run a CUDA application:

1. Ensure you have a CUDA-capable GPU
2. Install the appropriate NVIDIA driver
3. Execute the compiled binary:
   \`\`\`bash
   ./my_cuda_program
   \`\`\`

## Environment Variables

Several environment variables can control CUDA runtime behavior:

- **CUDA_VISIBLE_DEVICES**: Control which GPUs are visible to the application
  \`\`\`bash
  CUDA_VISIBLE_DEVICES=0,1 ./my_cuda_program  # Use only GPUs 0 and 1
  \`\`\`

- **CUDA_LAUNCH_BLOCKING**: Force synchronous kernel launches (useful for debugging)
  \`\`\`bash
  CUDA_LAUNCH_BLOCKING=1 ./my_cuda_program
  \`\`\`

## Profiling Tools

NVIDIA provides several tools for profiling CUDA applications:

### 1. Nsight Systems

A system-wide performance analysis tool:
\`\`\`bash
nsys profile ./my_cuda_program
\`\`\`

### 2. Nsight Compute

A detailed kernel performance analysis tool:
\`\`\`bash
ncu ./my_cuda_program
\`\`\`

### 3. nvprof (Legacy)

The original CUDA profiler (deprecated but still useful):
\`\`\`bash
nvprof ./my_cuda_program
\`\`\`

## Memory Checking

Use CUDA's memory checking tool to detect memory errors:
\`\`\`bash
cuda-memcheck ./my_cuda_program
\`\`\`

## Performance Optimization Tips

1. **Maximize occupancy**: Balance resources per thread
2. **Optimize memory access**: Ensure coalesced access patterns
3. **Minimize host-device transfers**: Keep data on the GPU
4. **Use asynchronous operations**: Overlap computation and data transfer
5. **Monitor resource usage**: Check for bottlenecks using profiling tools
    `,
    code: `# Basic execution
./vector_add

# Specify visible GPUs
CUDA_VISIBLE_DEVICES=0 ./vector_add

# Force synchronous kernel launches (for debugging)
CUDA_LAUNCH_BLOCKING=1 ./vector_add

# Profile with Nsight Systems
nsys profile -o profile_report ./vector_add

# Profile with Nsight Compute
ncu --set full -o kernel_analysis ./vector_add

# Profile with nvprof (legacy)
nvprof --log-file prof.log ./vector_add

# Check for memory errors
cuda-memcheck ./vector_add

# Analyze specific metrics
ncu --metrics sm__warps_active.avg,dram__bytes_read.sum ./vector_add

# Visual profiling
nsight-sys &  # Launch Nsight Systems UI
nsight-compute &  # Launch Nsight Compute UI`,
  },
]

export default function CudaCompilerGuide() {
  const [currentStep, setCurrentStep] = useState(0)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(cudaCompilerSteps[currentStep].code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const nextStep = () => {
    if (currentStep < cudaCompilerSteps.length - 1) {
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
              <CardTitle className="text-green-400">{cudaCompilerSteps[currentStep].title}</CardTitle>
              <CardDescription>CUDA Compilation and Execution</CardDescription>
            </div>
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {cudaCompilerSteps.length}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="content">
            <TabsList className="grid w-full grid-cols-2 bg-gray-900">
              <TabsTrigger value="content">Content</TabsTrigger>
              <TabsTrigger value="code">Command Examples</TabsTrigger>
            </TabsList>
            <TabsContent value="content" className="mt-4">
              <div className="bg-gray-900 p-4 rounded-md overflow-y-auto max-h-[500px]">
                <div className="prose prose-invert max-w-none">
                  <div
                    dangerouslySetInnerHTML={{
                      __html: cudaCompilerSteps[currentStep].content
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
                <pre className="bg-black p-4 rounded-md overflow-x-auto text-sm text-green-400 max-h-[500px] overflow-y-auto font-mono">
                  {cudaCompilerSteps[currentStep].code}
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
            disabled={currentStep === cudaCompilerSteps.length - 1}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
