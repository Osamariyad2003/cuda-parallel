"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Send, Bot, User, X, Minimize, Maximize, Mic, MicOff } from "lucide-react"
import { deepSpeechService } from "@/services/deepSpeechService"

// Define the message type
type Message = {
  id: string
  text: string
  sender: "user" | "bot"
  timestamp: Date
}

// CUDA knowledge base for the chatbot
const cudaKnowledgeBase = [
  {
    keywords: ["what", "cuda", "is"],
    response:
      "CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of the GPU.",
  },
  {
    keywords: ["thread", "block", "grid"],
    response:
      "In CUDA, threads are organized into blocks, and blocks are organized into grids. This hierarchy allows for efficient parallel execution. A thread is the smallest unit of execution, a block is a group of threads that can cooperate, and a grid is a collection of blocks.",
  },
  {
    keywords: ["shared", "memory"],
    response:
      "Shared memory in CUDA is a fast, on-chip memory that is shared among all threads in a block. It's much faster than global memory and is used for inter-thread communication and data reuse within a block.",
  },
  {
    keywords: ["warp"],
    response:
      "A warp in CUDA is a group of 32 threads that execute in lockstep (SIMT - Single Instruction, Multiple Thread). All threads in a warp execute the same instruction at the same time.",
  },
  {
    keywords: ["kernel"],
    response:
      "A CUDA kernel is a function that runs on the GPU. It's defined using the __global__ specifier and is called from the host (CPU) but executed on the device (GPU) by many threads in parallel.",
  },
  {
    keywords: ["memory", "hierarchy"],
    response:
      "CUDA has a memory hierarchy consisting of registers (fastest), shared memory, L1/L2 cache, and global memory (slowest). Understanding and optimizing memory access patterns is crucial for performance.",
  },
  {
    keywords: ["atomic"],
    response:
      "Atomic operations in CUDA ensure that read-modify-write operations on memory are performed without interference from other threads. They're useful for updating shared variables safely but can impact performance due to serialization.",
  },
  {
    keywords: ["occupancy"],
    response:
      "Occupancy in CUDA refers to the ratio of active warps to the maximum number of warps supported on a streaming multiprocessor (SM). Higher occupancy can help hide memory and instruction latency.",
  },
  {
    keywords: ["bank", "conflict"],
    response:
      "Bank conflicts occur in CUDA shared memory when multiple threads in a warp access different addresses in the same memory bank, causing serialization of memory accesses. They can be avoided using padding or careful access patterns.",
  },
  {
    keywords: ["coalesced", "access"],
    response:
      "Coalesced memory access in CUDA occurs when threads in a warp access contiguous memory locations. This allows the hardware to combine multiple memory transactions into fewer, larger transactions, improving performance.",
  },
  {
    keywords: ["hello", "hi", "hey"],
    response: "Hello! I'm your CUDA assistant. How can I help you with CUDA programming today?",
  },
  {
    keywords: ["thank", "thanks"],
    response: "You're welcome! If you have any more questions about CUDA, feel free to ask.",
  },
]

// Function to find the best response from the knowledge base
function findResponse(query: string): string {
  const normalizedQuery = query.toLowerCase()

  // Check for exact matches first
  for (const entry of cudaKnowledgeBase) {
    if (entry.keywords.every((keyword) => normalizedQuery.includes(keyword))) {
      return entry.response
    }
  }

  // If no exact match, find the best partial match
  let bestMatch = {
    entry: cudaKnowledgeBase[cudaKnowledgeBase.length - 1],
    matchCount: 0,
  }

  for (const entry of cudaKnowledgeBase) {
    const matchCount = entry.keywords.filter((keyword) => normalizedQuery.includes(keyword)).length

    if (matchCount > bestMatch.matchCount) {
      bestMatch = { entry, matchCount }
    }
  }

  // Return default response if no good match
  if (bestMatch.matchCount === 0) {
    return "I'm not sure about that. Could you ask something specific about CUDA programming?"
  }

  return bestMatch.entry.response
}

// Add types for speech recognition
type AudioState = {
  isRecording: boolean
  stream: MediaStream | null
}

export default function CudaChatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      text: "Hello! I'm your CUDA assistant. Ask me anything about CUDA programming!",
      sender: "bot",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isMinimized, setIsMinimized] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [audioState, setAudioState] = useState<AudioState>({
    isRecording: false,
    stream: null,
  })

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSendMessage = () => {
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: "user",
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")

    // Generate bot response
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: findResponse(input),
        sender: "bot",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, botResponse])
    }, 500)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSendMessage()
    }
  }

  // Function to handle speech recognition
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setAudioState({ isRecording: true, stream })

      // Create audio context and processor
      const audioContext = new AudioContext()
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)

      source.connect(processor)
      processor.connect(audioContext.destination)

      // Process audio data
      processor.onaudioprocess = async (e) => {
        const inputData = e.inputBuffer.getChannelData(0)
        try {
          // Send audio data to DeepSpeech for processing
          const transcription = await deepSpeechService.processAudio(inputData)
          
          if (transcription.trim()) {
            // Add transcribed text to input
            setInput((prev) => prev + " " + transcription)
          }
        } catch (error) {
          console.error("Error processing audio with DeepSpeech:", error)
        }
      }

    } catch (error) {
      console.error("Error accessing microphone:", error)
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (audioState.stream) {
      audioState.stream.getTracks().forEach(track => track.stop())
      setAudioState({ isRecording: false, stream: null })
    }
  }, [audioState.stream])

  return (
    <div className={`fixed bottom-4 right-4 z-50 transition-all duration-300 ${isMinimized ? "w-64" : "w-80 md:w-96"}`}>
      <Card className="border-green-400 shadow-lg">
        <CardHeader className="bg-gray-800 p-3 flex flex-row items-center justify-between">
          <CardTitle className="text-green-400 flex items-center text-base">
            <Bot className="mr-2 h-5 w-5" />
            CUDA Assistant
          </CardTitle>
          <div className="flex space-x-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-gray-400 hover:text-white"
              onClick={() => setIsMinimized(!isMinimized)}
            >
              {isMinimized ? <Maximize className="h-4 w-4" /> : <Minimize className="h-4 w-4" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-gray-400 hover:text-white"
              onClick={() =>
                setMessages([
                  {
                    id: "welcome",
                    text: "Hello! I'm your CUDA assistant. Ask me anything about CUDA programming!",
                    sender: "bot",
                    timestamp: new Date(),
                  },
                ])
              }
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>

        {!isMinimized && (
          <>
            <CardContent className="p-3 max-h-96 overflow-y-auto bg-gray-900">
              <div className="space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.sender === "user" ? "bg-green-600 text-white" : "bg-gray-800 text-gray-200"
                      }`}
                    >
                      <div className="flex items-center mb-1">
                        {message.sender === "bot" ? (
                          <Bot className="h-4 w-4 mr-1" />
                        ) : (
                          <User className="h-4 w-4 mr-1" />
                        )}
                        <span className="text-xs opacity-75">
                          {message.sender === "bot" ? "CUDA Assistant" : "You"}
                        </span>
                      </div>
                      <p className="text-sm">{message.text}</p>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            </CardContent>

            <CardFooter className="p-3 bg-gray-800 border-t border-gray-700">
              <div className="flex w-full space-x-2">
                <Input
                  placeholder="Ask about CUDA..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="bg-gray-700 border-gray-600 text-white"
                />
                <Button
                  onClick={() => {
                    if (audioState.isRecording) {
                      stopRecording()
                    } else {
                      startRecording()
                    }
                  }}
                  size="icon"
                  className={`${
                    audioState.isRecording ? "bg-red-500 hover:bg-red-600" : "bg-blue-500 hover:bg-blue-600"
                  }`}
                >
                  {audioState.isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </Button>
                <Button onClick={handleSendMessage} size="icon" className="bg-green-500 hover:bg-green-600">
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </CardFooter>
          </>
        )}
      </Card>
    </div>
  )
}
