import { NextResponse } from 'next/server'
import { DEEPSEEK_API_KEY } from '@/config/deepseek'

// Local knowledge base for fallback responses
const cudaKnowledgeBase = [
  {
    keywords: ["what", "cuda", "is"],
    response: "CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of the GPU."
  },
  {
    keywords: ["thread", "block", "grid"],
    response: "In CUDA, threads are organized into blocks, and blocks are organized into grids. This hierarchy allows for efficient parallel execution. A thread is the smallest unit of execution, a block is a group of threads that can cooperate, and a grid is a collection of blocks."
  },
  {
    keywords: ["shared", "memory"],
    response: "Shared memory in CUDA is a fast, on-chip memory that is shared among all threads in a block. It's much faster than global memory and is used for inter-thread communication and data reuse within a block."
  },
  {
    keywords: ["kernel"],
    response: "A CUDA kernel is a function that runs on the GPU. It's defined using the __global__ specifier and is called from the host (CPU) but executed on the device (GPU) by many threads in parallel."
  }
]

// Function to find the best matching response from local knowledge base
function findLocalResponse(query: string): string {
  const normalizedQuery = query.toLowerCase()
  
  // Check for exact matches first
  for (const entry of cudaKnowledgeBase) {
    if (entry.keywords.every(keyword => normalizedQuery.includes(keyword))) {
      return entry.response
    }
  }
  
  // If no exact match, find the best partial match
  let bestMatch = {
    entry: null as typeof cudaKnowledgeBase[0] | null,
    matchCount: 0
  }
  
  for (const entry of cudaKnowledgeBase) {
    const matchCount = entry.keywords.filter(keyword => normalizedQuery.includes(keyword)).length
    if (matchCount > bestMatch.matchCount) {
      bestMatch = { entry, matchCount }
    }
  }
  
  if (bestMatch.entry) {
    return bestMatch.entry.response
  }
  
  return "I'm not sure about that. Could you ask something specific about CUDA programming concepts, thread organization, memory types, or kernel execution?"
}

export async function POST(request: Request) {
  try {
    const { message } = await request.json()

    try {
      // First try the Deepseek API
      const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
        },
        body: JSON.stringify({
          model: 'deepseek-coder',
          messages: [
            {
              role: 'system',
              content: 'You are a CUDA programming expert assistant. Help users with CUDA-related questions, code examples, and best practices.',
            },
            {
              role: 'user',
              content: message,
            },
          ],
          temperature: 0.7,
          max_tokens: 1000,
        }),
      })

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`)
      }

      const data = await response.json()
      return NextResponse.json({ response: data.choices[0].message.content })
      
    } catch (apiError) {
      // If Deepseek API fails, fall back to local knowledge base
      console.warn('Deepseek API error, falling back to local knowledge base:', apiError)
      const localResponse = findLocalResponse(message)
      return NextResponse.json({ response: localResponse })
    }
    
  } catch (error) {
    console.error('Error in chat route:', error)
    return NextResponse.json(
      { response: "I apologize, but I'm having trouble processing your request. Please try asking about basic CUDA concepts." },
      { status: 200 } // Still return 200 to show a friendly message to the user
    )
  }
} 