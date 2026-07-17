import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import DrugSearch from '../components/DrugSearch'
import MethodSelector from '../components/MethodSelector'
import RecommendationTable from '../components/RecommendationTable'
import MetricsComparisonTable from '../components/MetricsComparisonTable'

const API_BASE = '/api'

export interface Drug {
  drugName: string
  condition?: string
  review?: string
  rating?: number
  usefulCount?: number
}

export interface Recommendation {
  drugName: string
  condition?: string
  similarityScore?: number
  predictedRating?: number
  contentScore?: number
  collabScore?: number
  hybridScore?: number
}

export interface Metrics {
  hybrid: {
    precision_at_10: number
    recall_at_10: number
    coverage: number
    execution_time: number
  }
  collaborative: {
    rmse: number
    mae: number
  }
  content: {
    status: string
  }
}

export interface ComparisonRow {
  metric: string
  hybrid: number
  content_based: number
  collaborative: number
}

function HomePage() {
  const [drugs, setDrugs] = useState<Drug[]>([])
  const [isLoadingDrugs, setIsLoadingDrugs] = useState(false)
  const [drugsError, setDrugsError] = useState<string | null>(null)

  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false)
  const [recommendationsError, setRecommendationsError] = useState<string | null>(null)

  const [, setMetrics] = useState<Metrics | null>(null)
  const [comparisonTable, setComparisonTable] = useState<ComparisonRow[]>([])
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false)
  const [metricsError, setMetricsError] = useState<string | null>(null)

  const [selectedMethod, setSelectedMethod] = useState<'content' | 'collaborative' | 'hybrid'>('content')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchHistory, setSearchHistory] = useState<string[]>([])

  const fetchDrugs = useCallback(async (limit: number = 50) => {
    setIsLoadingDrugs(true)
    setDrugsError(null)
    try {
      const response = await axios.get(`${API_BASE}/drugs?limit=${limit}`)
      setDrugs(response.data)
    } catch {
      setDrugsError('Failed to load drugs')
    } finally {
      setIsLoadingDrugs(false)
    }
  }, [])

  const fetchRecommendations = useCallback(async (method: string, query: string, history: string[] = []) => {
    if (!query.trim()) {
      setRecommendationsError('Please enter a search query')
      return
    }
    setIsLoadingRecommendations(true)
    setRecommendationsError(null)
    try {
      const response = await axios.post(`${API_BASE}/recommend`, {
        method: method,
        query: query.trim(),
        previous_searches: history,
      })
      if (response.data.status === 'success') {
        setRecommendations(response.data.recommendations)
      } else {
        setRecommendationsError('Failed to get recommendations')
      }
    } catch (err: any) {
      setRecommendationsError(err.response?.data?.detail || 'Failed to get recommendations')
    } finally {
      setIsLoadingRecommendations(false)
    }
  }, [])

  const fetchMetrics = useCallback(async () => {
    setIsLoadingMetrics(true)
    setMetricsError(null)
    try {
      const metricsResponse = await axios.get(`${API_BASE}/metrics`)
      setMetrics(metricsResponse.data.metrics)
      const comparisonResponse = await axios.get(`${API_BASE}/metrics/comparison`)
      if (comparisonResponse.data.comparison_table) {
        setComparisonTable(comparisonResponse.data.comparison_table)
      }
    } catch {
      setMetricsError('Failed to load metrics')
    } finally {
      setIsLoadingMetrics(false)
    }
  }, [])

  useEffect(() => {
    fetchDrugs()
    fetchMetrics()
  }, [fetchDrugs, fetchMetrics])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchQuery.trim()) return

    // Add current query to history (before searching)
    const newHistory = [...searchHistory]
    if (searchQuery.trim() !== '' && !newHistory.includes(searchQuery.trim())) {
      newHistory.push(searchQuery.trim())
    }
    setSearchHistory(newHistory)

    // Send recommendations with history
    fetchRecommendations(selectedMethod, searchQuery, newHistory.slice(0, -1))
  }

  const removeFromHistory = (drug: string) => {
    setSearchHistory(prev => prev.filter(d => d !== drug))
  }

  const clearHistory = () => {
    setSearchHistory([])
  }

  const drugNames = [...new Set(drugs.map(d => d.drugName))]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center h-9 w-9 rounded-xl bg-red-600 shadow-sm">
                <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-.659.464H8.129a2.25 2.25 0 01-.659-.464L5 14.5m14 0H5" />
                </svg>
              </div>
              <div>
                <h1 className="text-base font-bold text-gray-900">PharmaRec</h1>
                <p className="text-[11px] text-gray-400 font-medium">Drug Recommendation System</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={fetchMetrics} className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors" title="Refresh metrics">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.992 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search Section */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-8 slide-up">
          <div className="flex items-center gap-3 mb-5">
            <div className="h-8 w-8 rounded-lg bg-red-50 flex items-center justify-center">
              <svg className="h-4 w-4 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-gray-900">Find Recommendations</h2>
              <p className="text-xs text-gray-400">Search for a drug or enter a user ID</p>
            </div>
          </div>

          <form onSubmit={handleSearch}>
            <div className="flex flex-col sm:flex-row gap-3">
              {/* Search Input */}
              <div className="flex-1">
                <DrugSearch
                  value={searchQuery}
                  onChange={(value) => setSearchQuery(value)}
                  placeholder={selectedMethod === 'collaborative' ? 'Enter user ID (e.g. 100974)' : 'Enter drug name'}
                  drugNames={selectedMethod !== 'collaborative' ? drugNames : []}
                />
              </div>

              {/* Method Selector */}
              <div className="w-full sm:w-48">
                <MethodSelector
                  value={selectedMethod}
                  onChange={(value) => {
                    setSelectedMethod(value)
                    setSearchQuery('')
                  }}
                />
              </div>

              {/* Search Button */}
              <div className="sm:w-32">
                <button
                  type="submit"
                  disabled={isLoadingRecommendations || !searchQuery.trim()}
                  className="w-full inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white font-medium text-sm rounded-lg transition-all duration-150 shadow-sm hover:shadow focus:outline-none focus:ring-2 focus:ring-red-500/20 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed h-[42px]"
                >
                  {isLoadingRecommendations ? (
                    <>
                      <div className="spinner"></div>
                      <span>Searching</span>
                    </>
                  ) : (
                    'Search'
                  )}
                </button>
              </div>
            </div>
          </form>

          {/* Collaborative User ID Hint */}
          {selectedMethod === 'collaborative' && (
            <div className="mt-4 p-3 rounded-lg bg-red-50 border border-red-100">
              <div className="flex items-start gap-2">
                <svg className="h-4 w-4 text-red-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="text-xs font-medium text-red-700">Try these user IDs:</p>
                  <div className="flex flex-wrap gap-1.5 mt-1.5">
                    {['100974', '12636', '200864', '134763', '89889'].map(id => (
                      <button
                        key={id}
                        type="button"
                        onClick={() => setSearchQuery(id)}
                        className="px-2 py-0.5 rounded text-xs font-mono bg-white border border-red-200 text-red-700 hover:bg-red-50 transition-colors"
                      >
                        {id}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Search History */}
          {searchHistory.length > 0 && selectedMethod !== 'collaborative' && (
            <div className="mt-4 pt-4 border-t border-gray-100">
              <div className="flex items-center justify-between mb-2.5">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-xs font-medium text-gray-500">Search history ({searchHistory.length})</span>
                </div>
                <button
                  type="button"
                  onClick={clearHistory}
                  className="text-xs text-gray-400 hover:text-red-600 transition-colors"
                >
                  Clear
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {searchHistory.map((drug, idx) => (
                  <div
                    key={`${drug}-${idx}`}
                    className="group flex items-center gap-1.5 pl-2.5 pr-1.5 py-1 rounded-full bg-gray-100 hover:bg-red-50 border border-gray-200 hover:border-red-200 transition-colors"
                  >
                    <button
                      type="button"
                      onClick={() => {
                        setSearchQuery(drug)
                        setSelectedMethod('content')
                      }}
                      className="text-xs font-medium text-gray-700 group-hover:text-red-700 transition-colors"
                    >
                      {drug}
                    </button>
                    <button
                      type="button"
                      onClick={() => removeFromHistory(drug)}
                      className="text-gray-400 hover:text-red-600 transition-colors"
                      aria-label={`Remove ${drug}`}
                    >
                      <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
              {searchHistory.length > 0 && (
                <p className="text-[11px] text-gray-400 mt-2 flex items-center gap-1">
                  <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Results now combine your search history for better recommendations
                </p>
              )}
            </div>
          )}
        </div>

        {/* Error */}
        {recommendationsError && (
          <div className="mb-8 p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3 fade-in">
            <svg className="h-5 w-5 text-red-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
            <div>
              <p className="text-sm font-medium text-red-800">Search failed</p>
              <p className="text-sm text-red-600 mt-0.5">{recommendationsError}</p>
            </div>
          </div>
        )}

        {/* Loading */}
        {isLoadingRecommendations && (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-12 mb-8 text-center fade-in">
            <div className="spinner-lg mx-auto mb-4"></div>
            <p className="text-sm font-medium text-gray-500">Finding recommendations...</p>
          </div>
        )}

        {/* Results */}
        {!isLoadingRecommendations && recommendations.length > 0 && (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm mb-8 fade-in">
            <div className="p-5 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg bg-red-50 flex items-center justify-center">
                    <svg className="h-4 w-4 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
                    </svg>
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold text-gray-900">
                      Results for "<span className="text-red-600">{searchQuery}</span>"
                    </h2>
                    <p className="text-xs text-gray-400">{recommendations.length} recommendations found</p>
                  </div>
                </div>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200 capitalize">{selectedMethod}</span>
              </div>
            </div>
            <RecommendationTable recommendations={recommendations} method={selectedMethod} />
          </div>
        )}

        {/* Empty State */}
        {!isLoadingRecommendations && recommendations.length === 0 && !recommendationsError && (
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-12 mb-8 text-center">
            <div className="mx-auto h-12 w-12 rounded-2xl bg-gray-100 flex items-center justify-center mb-4">
              <svg className="h-6 w-6 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-500">Search for a drug to get started</p>
            <p className="text-xs text-gray-400 mt-1">Try searching for a drug name or user ID above</p>
          </div>
        )}

        {/* Bottom Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Metrics */}
          <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm">
            <div className="p-5 border-b border-gray-100">
              <div className="flex items-center gap-3">
                <div className="h-8 w-8 rounded-lg bg-gray-100 flex items-center justify-center">
                  <svg className="h-4 w-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-gray-900">Algorithm Comparison</h2>
                  <p className="text-xs text-gray-400">Performance metrics across methods</p>
                </div>
              </div>
            </div>
            <div className="p-5">
              {metricsError && (
                <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-100">
                  <p className="text-sm text-red-700">{metricsError}</p>
                </div>
              )}
              <MetricsComparisonTable comparisonData={comparisonTable} isLoading={isLoadingMetrics} />
            </div>
          </div>

          {/* Popular Drugs */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
            <div className="p-5 border-b border-gray-100">
              <div className="flex items-center gap-3">
                <div className="h-8 w-8 rounded-lg bg-gray-100 flex items-center justify-center">
                  <svg className="h-4 w-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.361-6.867 8.21 8.21 0 003 2.48z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18a3.75 3.75 0 00.495-7.467 5.99 5.99 0 00-1.925 3.546 5.974 5.974 0 01-2.133-1A3.75 3.75 0 0012 18z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-gray-900">Popular Drugs</h2>
                  <p className="text-xs text-gray-400">Click to search</p>
                </div>
              </div>
            </div>
            <div className="p-4">
              {isLoadingDrugs ? (
                <div className="flex items-center justify-center py-10">
                  <div className="spinner"></div>
                </div>
              ) : drugsError ? (
                <div className="p-3 rounded-lg bg-red-50 border border-red-100">
                  <p className="text-sm text-red-700">{drugsError}</p>
                </div>
              ) : (
                <div className="space-y-1">
                  {drugs.slice(0, 10).map((drug, index) => (
                    <button
                      key={index}
                      onClick={() => { setSearchQuery(drug.drugName); setSelectedMethod('content') }}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-gray-50 transition-colors group"
                    >
                      <span className="flex-shrink-0 h-6 w-6 rounded-md bg-red-50 flex items-center justify-center text-xs font-bold text-red-600 group-hover:bg-red-100 transition-colors">
                        {index + 1}
                      </span>
                      <span className="text-sm font-medium text-gray-700 group-hover:text-red-600 transition-colors truncate">
                        {drug.drugName}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white mt-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className="font-medium text-gray-500">PharmaRec</span>
              <span>·</span>
              <span>Comparative Drug Recommendation System</span>
            </div>
            <div className="flex items-center gap-4 text-xs text-gray-400">
              <span>Content-Based</span>
              <span className="text-gray-300">|</span>
              <span>Collaborative Filtering</span>
              <span className="text-gray-300">|</span>
              <span>Hybrid</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default HomePage
