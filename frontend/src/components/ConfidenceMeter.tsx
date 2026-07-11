import React from 'react'
import { RadialBarChart, RadialBar, ResponsiveContainer } from 'recharts'

interface Props {
  confidence: number
  verdict: 'yes' | 'no' | 'uncertain'
}

const VERDICT_CONFIG = {
  yes: { label: 'AI Generated', color: '#f43f5e', bg: 'bg-rose-950/40', border: 'border-rose-800', text: 'text-rose-400' },
  no: { label: 'Likely Real', color: '#22c55e', bg: 'bg-green-950/40', border: 'border-green-800', text: 'text-green-400' },
  uncertain: { label: 'Uncertain', color: '#f59e0b', bg: 'bg-amber-950/40', border: 'border-amber-800', text: 'text-amber-400' },
}

export default function ConfidenceMeter({ confidence, verdict }: Props) {
  const cfg = VERDICT_CONFIG[verdict]
  const pct = Math.round(confidence * 100)
  const data = [{ value: pct, fill: cfg.color }]

  return (
    <div className={`rounded-2xl border p-6 ${cfg.bg} ${cfg.border} flex flex-col items-center gap-2`}>
      <div className="relative w-40 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="70%"
            outerRadius="100%"
            startAngle={90}
            endAngle={-270}
            data={data}
            barSize={14}
          >
            <RadialBar
              background={{ fill: '#1e293b' }}
              dataKey="value"
              cornerRadius={8}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-white">{pct}%</span>
          <span className="text-xs text-slate-400">confidence</span>
        </div>
      </div>

      <div className={`text-lg font-semibold ${cfg.text}`}>{cfg.label}</div>
    </div>
  )
}
