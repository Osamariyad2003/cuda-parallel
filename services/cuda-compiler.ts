import { exec } from 'child_process';
import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';

interface CompilationResult {
  success: boolean;
  output: string;
  error?: string;
}

export async function compileCudaCode(code: string): Promise<CompilationResult> {
  const tempDir = process.env.TEMP || '/tmp';
  const sourceFile = join(tempDir, `cuda_program_${Date.now()}.cu`);
  const outputFile = join(tempDir, `cuda_program_${Date.now()}.exe`);

  try {
    // Write CUDA code to temporary file
    await writeFile(sourceFile, code);

    // Compile the code
    const compilationResult = await new Promise<CompilationResult>((resolve) => {
      exec(`nvcc "${sourceFile}" -o "${outputFile}"`, (compileError, stdout, stderr) => {
        if (compileError) {
          resolve({
            success: false,
            output: '',
            error: stderr || compileError.message
          });
          return;
        }

        // Run the compiled program
        exec(`"${outputFile}"`, (runError, runStdout, runStderr) => {
          resolve({
            success: !runError,
            output: runStdout,
            error: runError ? runStderr : undefined
          });
        });
      });
    });

    // Clean up temporary files
    await Promise.all([
      unlink(sourceFile).catch(() => {}),
      unlink(outputFile).catch(() => {})
    ]);

    return compilationResult;
  } catch (error) {
    return {
      success: false,
      output: '',
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
} 