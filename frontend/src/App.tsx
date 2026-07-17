import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import HomePage from './pages/Home'

const API_BASE = '/api'

function App() {
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    let cancelled = false

    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE}/health`, { timeout: 8000 })
        if (cancelled) return
        if (response.data.status === 'healthy') {
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          setState('ready')
        }
      } catch {
        if (cancelled) return
        setRetryCount(prev => {
          const next = prev + 1
          if (next > 12) {
            setErrorMsg('Backend is not responding. Make sure the API server is running.')
            setState('error')
          }
          return next
        })
      }
    }

    checkHealth()
    intervalRef.current = setInterval(checkHealth, 5000)

    return () => {
      cancelled = true
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  if (state === 'error') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="card p-8 text-center">
            <div className="mx-auto h-14 w-14 rounded-2xl bg-brand-50 flex items-center justify-center mb-6">
              <svg className="h-7 w-7 text-brand-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
              </svg>
            </div>
            <h1 className="text-xl font-semibold text-gray-900 mb-2">Unable to connect</h1>
            <p className="text-sm text-gray-500 mb-8 leading-relaxed">{errorMsg}</p>
            <div className="text-left space-y-4 mb-8">
              <div className="p-4 rounded-lg bg-gray-50 border border-gray-100">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Backend</p>
                <code className="block text-sm text-gray-700 font-mono">
                  cd backend && source venv/bin/activate && uvicorn app.main:app --reload
                </code>
              </div>
              <div className="p-4 rounded-lg bg-gray-50 border border-gray-100">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Frontend</p>
                <code className="block text-sm text-gray-700 font-mono">
                  cd frontend && npm run dev
                </code>
              </div>
            </div>
            <button
              onClick={() => { setState('loading'); setRetryCount(0) }}
              className="btn-primary w-full"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (state === 'ready') {
    return <HomePage />
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="spinner-lg mx-auto mb-4"></div>
        <p className="text-sm font-medium text-gray-500">Connecting to API...</p>
        {retryCount > 3 && (
          <p className="text-xs text-gray-400 mt-2">Loading dataset for the first time, this may take a moment</p>
        )}
      </div>
    </div>
  )
}

export default App
