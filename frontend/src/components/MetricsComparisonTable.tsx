import { ComparisonRow } from '../pages/Home'

interface MetricsComparisonTableProps {
  comparisonData: ComparisonRow[]
  isLoading: boolean
}

function MetricsComparisonTable({ comparisonData, isLoading }: MetricsComparisonTableProps) {
  const formatMetricName = (metric: string) => {
    switch (metric) {
      case 'precision_at_10': return 'Precision@10'
      case 'recall_at_10': return 'Recall@10'
      case 'coverage': return 'Coverage'
      case 'execution_time': return 'Exec Time (s)'
      default: return metric
    }
  }

  const formatValue = (_metric: string, value: number) => value.toFixed(4)

  const getBarWidth = (value: number, metric: string) => {
    if (metric === 'execution_time') return `${Math.min(value * 5000, 100)}%`
    return `${Math.min(value * 100, 100)}%`
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="spinner-lg"></div>
      </div>
    )
  }

  if (comparisonData.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="mx-auto h-12 w-12 rounded-2xl bg-gray-100 flex items-center justify-center mb-4">
          <svg className="h-6 w-6 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75z" />
          </svg>
        </div>
        <p className="text-sm font-medium text-gray-400">No metrics available</p>
        <p className="text-xs text-gray-400 mt-1">Run some searches first</p>
      </div>
    )
  }

  const columns = [
    { key: 'hybrid' as const, label: 'Hybrid', color: 'bg-red-500' },
    { key: 'content_based' as const, label: 'Content', color: 'bg-gray-900' },
    { key: 'collaborative' as const, label: 'Collaborative', color: 'bg-gray-400' },
  ]

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-4 mb-6">
        {columns.map(col => (
          <div key={col.key} className="flex items-center gap-1.5">
            <div className={`h-2 w-2 rounded-full ${col.color}`}></div>
            <span className="text-xs font-medium text-gray-500">{col.label}</span>
          </div>
        ))}
      </div>

      <div className="space-y-5">
        {comparisonData.map((row, index) => (
          <div key={index}>
            <p className="text-sm font-medium text-gray-700 mb-2">{formatMetricName(row.metric)}</p>
            <div className="space-y-1.5">
              {columns.map(col => {
                const val = row[col.key]
                const display = val !== undefined && val !== null ? formatValue(row.metric, val) : 'N/A'
                const barWidth = val !== undefined && val !== null ? getBarWidth(val, row.metric) : '0%'
                return (
                  <div key={col.key} className="flex items-center gap-3">
                    <span className="text-xs text-gray-400 w-20 shrink-0">{col.label}</span>
                    <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${col.color} transition-all duration-500`} style={{ width: barWidth }} />
                    </div>
                    <span className="text-xs font-mono text-gray-500 w-16 text-right tabular-nums">{display}</span>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default MetricsComparisonTable
