import CudaCodeSimulator from "@/components/cuda-code-simulator"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"
import CudaChatbot from "@/components/cuda-chatbot"

export default function CodeSimulatorPage() {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-8">
          <Link href="/" className="text-green-400 hover:text-green-300 flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </div>

        <div className="text-center mb-12">
          <h1 className="text-3xl font-extrabold text-green-400 sm:text-4xl">CUDA Code Simulator</h1>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-300">
            Try out CUDA code examples and see the simulated output
          </p>
        </div>

        <div className="mb-8">
          <CudaCodeSimulator />
        </div>

        <div className="bg-gray-800 rounded-lg p-6 mt-8">
          <h2 className="text-xl font-bold text-green-400 mb-4">About the Simulator</h2>
          <p className="text-gray-300 mb-4">
            This is a simulated CUDA code execution environment. It doesn't actually compile or run CUDA code (which
            would require NVIDIA GPU drivers), but instead shows you predefined outputs for educational purposes.
          </p>
          <p className="text-gray-300">
            Use this simulator to understand how CUDA code works and what output to expect when running on a real GPU.
            For actual CUDA development, you'll need to install the CUDA Toolkit on a system with an NVIDIA GPU.
          </p>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 mt-8">
          <h2 className="text-xl font-bold text-green-400 mb-4">Official CUDA Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
            <a
              href="https://developer.nvidia.com/cuda-zone"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gray-900 p-4 rounded-lg hover:bg-gray-700 transition-colors flex flex-col items-center"
            >
              <img
                src="https://sjc.microlink.io/6NPrACaXpxaD2B3fb5WndDSLfE5YHkoWvfxcDW7WhwBsCM13zLCB8YhTbfaW5fXpiOdoTs5km4wQbTtyoyBQ_Q.jpeg"
                alt="NVIDIA CUDA Zone"
                className="w-full h-32 object-cover object-top rounded mb-3"
              />
              <h3 className="font-medium text-green-400">CUDA Zone</h3>
            </a>
            <a
              href="https://developer.nvidia.com/cuda-toolkit"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gray-900 p-4 rounded-lg hover:bg-gray-700 transition-colors flex flex-col items-center"
            >
              <img
                src="https://sjc.microlink.io/IILV5frtDD0qgj8nUb_wcHQa9nsxsrTPgVS9LQQvhLsUScNPV7KH9KEv8CtxmgpRrbOwICdrSadKSILTVW7vdg.jpeg"
                alt="NVIDIA CUDA Toolkit"
                className="w-full h-32 object-cover object-top rounded mb-3"
              />
              <h3 className="font-medium text-green-400">CUDA Toolkit</h3>
            </a>
            <a
              href="https://www.google.com/search?q=cuda&rlz=1C1GCEU_enJO1129JO1129&oq=cuda&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MggIARBFGCcYOzIHCAIQABiABDIHCAMQABiABDIGCAQQRRg8MgYIBRBFGDwyBggGEEUYPDIGCAcQRRg80gEIMTQ0N2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:a15d0016,vid:GmNkYayuaA4,st:0"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gray-900 p-4 rounded-lg hover:bg-gray-700 transition-colors flex flex-col items-center"
            >
              <img
                src="https://sjc.microlink.io/N1uqhNScr8JD2p8JAFsdwi2Ob5bA68u8TD7hWv60J6s1r8-W1LQabeokA0paJZFoWgL5wxJRADZW-FTv6fEP0w.jpeg"
                alt="CUDA Video Tutorial"
                className="w-full h-32 object-cover object-top rounded mb-3"
              />
              <h3 className="font-medium text-green-400">Video Tutorial</h3>
            </a>
          </div>
          <p className="text-gray-300 mt-6">
            For real CUDA development, visit these official NVIDIA resources to download the CUDA Toolkit, access
            documentation, and join the developer community.
          </p>
        </div>
      </div>

      {/* Add the chatbot to the code simulator page */}
      <CudaChatbot />
    </div>
  )
}
