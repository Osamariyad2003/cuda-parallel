"use client"

import type React from "react"

import { useState, useEffect } from "react"
import ErrorBoundary from "./error-boundary"

interface SafeCanvasWrapperProps {
  children: React.ReactNode
  fallback: React.ReactNode
  className?: string
}

export default function SafeCanvasWrapper({ children, fallback, className = "" }: SafeCanvasWrapperProps) {
  const [isClient, setIsClient] = useState(false)
  const [hasWebGL, setHasWebGL] = useState(false)

  useEffect(() => {
    setIsClient(true)

    // Check for WebGL support
    try {
      const canvas = document.createElement("canvas")
      const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl")
      setHasWebGL(!!gl)
    } catch (e) {
      setHasWebGL(false)
      console.error("WebGL not supported:", e)
    }
  }, [])

  if (!isClient) {
    // Server-side rendering or initial load
    return <div className={className}>{fallback}</div>
  }

  if (!hasWebGL) {
    // WebGL not supported
    return <div className={className}>{fallback}</div>
  }

  return (
    <ErrorBoundary fallback={<div className={className}>{fallback}</div>}>
      <div className={className}>{children}</div>
    </ErrorBoundary>
  )
}
