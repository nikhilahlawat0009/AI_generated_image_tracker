import React from 'react'
import { Brain, Waves, FileSearch } from 'lucide-react'

interface Props {
  scores: { model: number; frequency: number; metadata: number }
}

const DETECTORS = [
  {
    key: 'model' as const,
    label: 'ML Classifier',
    description: 'ViT-based model trained on real vs AI images',
    Icon: Brain,
    weight: '50%',
  },
  {
    key: 'frequency' as const,
    label: 'Frequency Analysis',
    description: 'FFT spectral artifacts & GAN fingerprints',
    Icon: Waves,
    weight: '25%',
  },
  {
    key: 'metadata' as const,
    label: 'Metadata Forensics',
    description: 'EXIF data & embedded generator signatures',
    Icon: FileSearch,
    weight: '25%',
  },
]

function scoreColor(score: number) {
  if (score >= 0.7) return 'bg-rose-500'
  if (score >= 0.45) return 'bg-amber-400'
  return 'bg-green-500'
}

export default function DetectorBreakdown({ scores }: Props) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Detector Breakdown</h3>
      {DETECTORS.map(({ key, label, description, Icon, weight }) => {
        const score = scores[key]
        const pct = Math.round(score * 100)
        return (
          <div key={key} className="bg-slate-900 rounded-xl p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Icon className="w-4 h-4 text-slate-400" />
                <span className="text-sm font-medium text-slate-200">{label}</span>
                <span className="text-xs text-slate-600 bg-slate-800 px-1.5 py-0.5 rounded">{weight}</span>
              </div>
              <span className="text-sm font-mono font-semibold text-slate-300">{pct}%</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ${scoreColor(score)}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="text-xs text-slate-500">{description}</p>
          </div>
        )
      })}
    </div>
  )
}
