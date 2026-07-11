import React, { useState } from 'react'
import { ScanSearch, AlertCircle } from 'lucide-react'
import ImageUploader from './components/ImageUploader'
import ConfidenceMeter from './components/ConfidenceMeter'
import DetectorBreakdown from './components/DetectorBreakdown'
import EvidencePanel from './components/EvidencePanel'
import AgentReport from './components/AgentReport'
import HistorySidebar from './components/HistorySidebar'
import type { AnalysisResult } from './types'

export default function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState('')
  const [historyRefresh, setHistoryRefresh] = useState(0)

  function handleResult(data: unknown) {
    if (!data) { setResult(null); return }
    setResult(data as AnalysisResult)
    setHistoryRefresh((n) => n + 1)
  }

  async function loadFromHistory(id: string) {
    try {
      const res = await fetch(`/api/analysis/${id}`)
      const data = await res.json()
      setResult(data as AnalysisResult)
    } catch {
      setError('Failed to load analysis')
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-sky-600 flex items-center justify-center">
            <ScanSearch className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight">AI Image Tracker</h1>
            <p className="text-xs text-slate-500">ML · Frequency Analysis · Metadata Forensics · Claude Agent</p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left column: upload + history */}
          <div className="space-y-6">
            <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-5 space-y-4">
              <h2 className="text-sm font-semibold text-slate-300">Upload Image</h2>
              <ImageUploader onResult={handleResult} onError={setError} />
              {error && (
                <div className="flex items-center gap-2 bg-rose-950/40 border border-rose-800 rounded-xl p-3">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                  <p className="text-sm text-rose-300">{error}</p>
                </div>
              )}
            </div>

            <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-5">
              <HistorySidebar refreshTrigger={historyRefresh} onSelect={loadFromHistory} />
            </div>
          </div>

          {/* Right: results */}
          <div className="lg:col-span-2 space-y-5">
            {!result ? (
              <div className="h-full min-h-64 flex items-center justify-center rounded-2xl border border-dashed border-slate-800">
                <div className="text-center space-y-2">
                  <ScanSearch className="w-10 h-10 text-slate-700 mx-auto" />
                  <p className="text-slate-600">Upload an image to start analysis</p>
                </div>
              </div>
            ) : (
              <>
                {/* Top row: meter + breakdown */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <ConfidenceMeter
                    confidence={result.confidence}
                    verdict={result.is_ai_generated}
                  />
                  <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-5">
                    <DetectorBreakdown scores={result.scores} />
                  </div>
                </div>

                {/* Evidence */}
                <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-5 space-y-3">
                  <h3 className="text-sm font-semibold text-slate-300">Evidence & Explanation</h3>
                  <EvidencePanel
                    flags={result.evidence_flags}
                    explanation={result.verdict_explanation}
                  />
                </div>

                {/* Agent report */}
                <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-5">
                  <AgentReport report={result.agent_report} />
                </div>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
