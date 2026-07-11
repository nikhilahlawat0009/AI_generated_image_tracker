import React from 'react'
import { AlertTriangle, CheckCircle, Info } from 'lucide-react'

interface Props {
  flags: string[]
  explanation: string
}

export default function EvidencePanel({ flags, explanation }: Props) {
  return (
    <div className="space-y-4">
      <div className="bg-slate-900 rounded-xl p-4 space-y-2">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-sky-400 mt-0.5 shrink-0" />
          <p className="text-sm text-slate-300">{explanation}</p>
        </div>
      </div>

      {flags.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Evidence Flags</h3>
          {flags.map((flag, i) => (
            <div key={i} className="flex items-start gap-2 bg-slate-900 rounded-xl p-3">
              <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
              <p className="text-sm text-slate-300">{flag}</p>
            </div>
          ))}
        </div>
      )}

      {flags.length === 0 && (
        <div className="flex items-center gap-2 bg-slate-900 rounded-xl p-3">
          <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />
          <p className="text-sm text-slate-400">No specific anomalies flagged by automated detectors.</p>
        </div>
      )}
    </div>
  )
}
