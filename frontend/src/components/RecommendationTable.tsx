import { Recommendation } from '../pages/Home'

interface RecommendationTableProps {
  recommendations: Recommendation[]
  method: 'content' | 'collaborative' | 'hybrid'
}

function RecommendationTable({ recommendations, method }: RecommendationTableProps) {
  const getColumns = () => {
    switch (method) {
      case 'content':
        return [
          { key: 'rank', header: '#' },
          { key: 'drugName', header: 'Drug Name' },
          { key: 'condition', header: 'Condition' },
          { key: 'similarityScore', header: 'Similarity Score' }
        ]
      case 'collaborative':
        return [
          { key: 'rank', header: '#' },
          { key: 'drugName', header: 'Drug Name' },
          { key: 'predictedRating', header: 'Predicted Rating' }
        ]
      case 'hybrid':
        return [
          { key: 'rank', header: '#' },
          { key: 'drugName', header: 'Drug Name' },
          { key: 'contentScore', header: 'Content Score' },
          { key: 'collabScore', header: 'Collab Score' },
          { key: 'hybridScore', header: 'Hybrid Score' }
        ]
      default:
        return [
          { key: 'rank', header: '#' },
          { key: 'drugName', header: 'Drug Name' }
        ]
    }
  }

  const columns = getColumns()

  const ScoreBar = ({ value, max = 1, color = 'bg-red-500' }: { value: number; max?: number; color?: string }) => {
    const pct = Math.round((value / max) * 100)
    return (
      <div className="flex items-center gap-2.5">
        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div className={`h-full rounded-full ${color} transition-all duration-500`} style={{ width: `${Math.min(pct, 100)}%` }} />
        </div>
        <span className="text-xs font-mono text-gray-500 w-14 text-right tabular-nums">{value.toFixed(4)}</span>
      </div>
    )
  }

  const formatValue = (rec: Recommendation, key: string, index: number) => {
    switch (key) {
      case 'rank':
        return (
          <span className="inline-flex items-center justify-center h-6 w-6 rounded-md bg-gray-100 text-xs font-bold text-gray-500">
            {index + 1}
          </span>
        )
      case 'similarityScore':
      case 'contentScore':
      case 'hybridScore': {
        const value = rec[key as keyof Recommendation]
        if (value !== undefined && value !== null) {
          const num = typeof value === 'number' ? value : parseFloat(value as string)
          return <ScoreBar value={num} color="bg-red-500" />
        }
        return <span className="text-gray-300">—</span>
      }
      case 'predictedRating':
      case 'collabScore': {
        const value = rec[key as keyof Recommendation]
        if (value !== undefined && value !== null) {
          const num = typeof value === 'number' ? value : parseFloat(value as string)
          const max = key === 'predictedRating' ? 10 : 1
          return <ScoreBar value={num} max={max} color="bg-gray-900" />
        }
        return <span className="text-gray-300">—</span>
      }
      default: {
        const val = rec[key as keyof Recommendation]
        return val !== undefined && val !== null ? (
          <span className="text-gray-700">{String(val)}</span>
        ) : (
          <span className="text-gray-300">—</span>
        )
      }
    }
  }

  return (
    <div className="overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="bg-gray-50">
            {columns.map((col) => (
              <th key={col.key} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-200">{col.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {recommendations.map((rec, index) => (
            <tr key={index} className="border-b border-gray-100 last:border-0 hover:bg-gray-50/50 transition-colors">
              {columns.map((col) => (
                <td key={col.key} className="px-4 py-3.5 text-sm text-gray-700">{formatValue(rec, col.key, index)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default RecommendationTable
