interface MethodSelectorProps {
  value: 'content' | 'collaborative' | 'hybrid'
  onChange: (value: 'content' | 'collaborative' | 'hybrid') => void
}

const methods = [
  { id: 'content' as const, name: 'Content-Based', description: 'Similar drugs by text analysis' },
  { id: 'collaborative' as const, name: 'Collaborative', description: 'Top-rated drugs for the same condition' },
  { id: 'hybrid' as const, name: 'Hybrid', description: 'Combined approach' },
]

function MethodSelector({ value, onChange }: MethodSelectorProps) {
  const active = methods.find(m => m.id === value)

  return (
    <div className="w-full">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as 'content' | 'collaborative' | 'hybrid')}
        className="w-full px-4 py-2.5 bg-white border border-gray-200 rounded-lg text-sm text-gray-900 transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-red-500/10 focus:border-red-500 focus:shadow-sm appearance-none cursor-pointer h-[42px]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`,
          backgroundPosition: 'right 0.5rem center',
          backgroundRepeat: 'no-repeat',
          backgroundSize: '1.5em 1.5em',
          paddingRight: '2.5rem',
        }}
      >
        {methods.map((method) => (
          <option key={method.id} value={method.id}>{method.name}</option>
        ))}
      </select>
      <div className="flex items-center gap-1.5 mt-2 text-xs text-gray-400">
        <svg className="h-3 w-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>{active?.description}</span>
      </div>
    </div>
  )
}

export default MethodSelector
