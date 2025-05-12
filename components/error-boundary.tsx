"use client"

import { Component, type ErrorInfo, type ReactNode } from "react"

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(_: Error): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Error caught by ErrorBoundary:", error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="p-4 bg-gray-800 rounded-lg border border-gray-700 text-center">
            <h3 className="text-lg font-medium text-red-400 mb-2">Visualization Error</h3>
            <p className="text-gray-300">
              There was an error loading this visualization. Please try refreshing the page.
            </p>
          </div>
        )
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
