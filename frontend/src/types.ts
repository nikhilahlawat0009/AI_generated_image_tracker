export interface AnalysisResult {
  id: string
  filename: string
  is_ai_generated: 'yes' | 'no' | 'uncertain'
  confidence: number
  scores: {
    model: number
    frequency: number
    metadata: number
  }
  verdict_explanation: string
  evidence_flags: string[]
  agent_report: string
  uploaded_at: string
}

export interface HistoryItem {
  id: string
  filename: string
  is_ai_generated: 'yes' | 'no' | 'uncertain'
  confidence: number
  uploaded_at: string
}
