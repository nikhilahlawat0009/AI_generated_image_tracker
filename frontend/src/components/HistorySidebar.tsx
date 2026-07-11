import React, { useEffect, useState } from 'react'
import { Clock, RefreshCw } from 'lucide-react'
import type { HistoryItem } from '../types'

const VERDICT_BADGE = {
  yes: 'bg-rose-900 text-rose-300',
  no: 'bg-green-900 text-green-300',
  uncertain: 'bg-amber-900 text-amber-300',
}
const VERDICT_LABEL = { yes: 'AI', no: 'Real', uncertain: '?' }

interface Props {
  refreshTrigger: number
  onSelect: (id: string) => void
}

export default function HistorySidebar({ refreshTrigger, onSelect }: Props) {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetch('/api/history')
      .then((r) => r.json())
      .then(setItems)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [refreshTrigger])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5" /> History
        </h3>
        {loading && <RefreshCw className="w-3.5 h-3.5 text-slate-600 animate-spin" />}
      </div>

      {items.length === 0 && !loading && (
        <p className="text-xs text-slate-600 text-center py-4">No analyses yet</p>
      )}

      <div className="space-y-2 max-h-80 overflow-y-auto scrollbar-thin">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className="w-full text-left bg-slate-900 hover:bg-slate-800 rounded-xl p-3 transition-colors"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-slate-300 truncate">{item.filename}</span>
              <span className={`text-xs font-bold px-1.5 py-0.5 rounded shrink-0 ${VERDICT_BADGE[item.is_ai_generated]}`}>
                {VERDICT_LABEL[item.is_ai_generated]}
              </span>
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className="text-xs text-slate-600">
                {new Date(item.uploaded_at).toLocaleDateString()}
              </span>
              <span className="text-xs text-slate-500">{Math.round(item.confidence * 100)}%</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
