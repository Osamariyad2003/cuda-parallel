import type { Metadata } from 'next'
import './globals.css'
import CudaChatbot from '@/components/cuda-chatbot'

export const metadata: Metadata = {
  title: 'CUDA Tutorial',
  description: 'Learn CUDA programming with interactive examples',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        {children}
        <CudaChatbot />
      </body>
    </html>
  )
}
