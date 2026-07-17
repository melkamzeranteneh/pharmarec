import { useState, useRef, useEffect } from 'react'

interface DrugSearchProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  drugNames?: string[]
}

function DrugSearch({ value, onChange, placeholder = 'Enter drug name or user ID', drugNames = [] }: DrugSearchProps) {
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (value && drugNames.length > 0) {
      const filtered = drugNames
        .filter(drug => drug.toLowerCase().includes(value.toLowerCase()) && drug.toLowerCase() !== value.toLowerCase())
        .slice(0, 5)
      setFilteredSuggestions(filtered)
      setShowSuggestions(filtered.length > 0 && value.length > 0)
    } else {
      setFilteredSuggestions([])
      setShowSuggestions(false)
    }
  }, [value, drugNames])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div className="relative" ref={containerRef}>
      <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
        <svg className="h-4 w-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
        </svg>
      </div>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full pl-10 pr-10 py-2.5 bg-white border border-gray-200 rounded-lg text-sm text-gray-900 placeholder:text-gray-400 transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-red-500/10 focus:border-red-500 focus:shadow-sm h-[42px]"
        onFocus={() => { if (value && filteredSuggestions.length > 0) setShowSuggestions(true) }}
      />
      {value && (
        <button
          onClick={() => { onChange(''); setShowSuggestions(false) }}
          className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-300 hover:text-gray-500 transition-colors"
          aria-label="Clear"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}

      {showSuggestions && filteredSuggestions.length > 0 && (
        <div className="absolute z-20 w-full mt-2 bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden fade-in">
          <div className="p-1">
            {filteredSuggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => { onChange(suggestion); setShowSuggestions(false) }}
                className="w-full flex items-center gap-3 px-3 py-2.5 text-left text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <svg className="h-4 w-4 text-gray-300 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                </svg>
                <span className="font-medium">{suggestion}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default DrugSearch
