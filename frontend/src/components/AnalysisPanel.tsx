import { Analysis } from '../pages/Home'

interface AnalysisPanelProps {
  analysis: Analysis | null
  isLoading: boolean
  error: string | null
}

function AnalysisPanel({ analysis, isLoading, error }: AnalysisPanelProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="spinner-lg"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-3 rounded-lg bg-red-50 border border-red-100">
        <p className="text-sm text-red-700">{error}</p>
      </div>
    )
  }

  if (!analysis) {
    return (
      <div className="text-center py-12">
        <p className="text-sm font-medium text-gray-400">No analysis available</p>
        <p className="text-xs text-gray-400 mt-1">Refresh metrics to generate the comparison</p>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-red-50 text-red-700 border border-red-200">
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 18.75h-9m9 0a3 3 0 013 3h-15a3 3 0 013-3m9 0v-3.375c0-.621-.503-1.125-1.125-1.125h-.871M7.5 18.75v-3.375c0-.621.504-1.125 1.125-1.125h.872m5.007 0H9.497m5.007 0a7.454 7.454 0 01-.982-3.172M9.497 14.25a7.454 7.454 0 00.981-3.172M5.25 4.236c-.982.143-1.954.317-2.916.52A6.003 6.003 0 007.73 9.728M5.25 4.236V4.5c0 2.108.966 3.99 2.48 5.228M5.25 4.236V2.721C7.456 2.41 9.71 2.25 12 2.25c2.291 0 4.545.16 6.75.47v1.516M7.73 9.728a6.726 6.726 0 002.748 1.35m8.272-6.842V4.5c0 2.108-.966 3.99-2.48 5.228m2.48-5.492a46.32 46.32 0 012.916.52 6.003 6.003 0 01-5.395 4.972m0 0a6.726 6.726 0 01-2.749 1.35m0 0a6.772 6.772 0 01-3.044 0" />
          </svg>
          Best Method: {analysis.winner}
        </span>
      </div>

      {/* Explanation paragraphs */}
      <div className="space-y-3">
        {analysis.explanation.map((para, i) => (
          <p key={i} className="text-sm text-gray-600 leading-relaxed">
            {para}
          </p>
        ))}
      </div>

      {/* Ranking table */}
      <div className="overflow-hidden rounded-lg border border-gray-100">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-3 py-2 text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Rank</th>
              <th className="px-3 py-2 text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Method</th>
              <th className="px-3 py-2 text-right text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Prec@10</th>
              <th className="px-3 py-2 text-right text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Recall@10</th>
              <th className="px-3 py-2 text-right text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Coverage</th>
            </tr>
          </thead>
          <tbody>
            {analysis.ranking.map((row, i) => (
              <tr key={row.method} className="border-t border-gray-100">
                <td className="px-3 py-2">
                  <span className={`inline-flex items-center justify-center h-5 w-5 rounded text-[11px] font-bold ${i === 0 ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-500'}`}>
                    {i + 1}
                  </span>
                </td>
                <td className="px-3 py-2 text-sm font-medium text-gray-700">{row.method}</td>
                <td className="px-3 py-2 text-right text-xs font-mono text-gray-500 tabular-nums">{row.precision_at_10.toFixed(4)}</td>
                <td className="px-3 py-2 text-right text-xs font-mono text-gray-500 tabular-nums">{row.recall_at_10.toFixed(4)}</td>
                <td className="px-3 py-2 text-right text-xs font-mono text-gray-500 tabular-nums">{row.coverage.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default AnalysisPanel
