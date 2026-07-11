import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Bot, ChevronDown, ChevronUp } from 'lucide-react'

interface Props {
  report: string
}

export default function AgentReport({ report }: Props) {
  const [expanded, setExpanded] = useState(true)

  return (
    <div className="bg-slate-900 rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded((e) => !e)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-800/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-sky-400" />
          <span className="text-sm font-semibold text-slate-200">Claude Forensic Agent Report</span>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-500" />
        )}
      </button>

      {expanded && (
        <div className="px-4 pb-4 border-t border-slate-800">
          <div className="prose prose-sm prose-invert max-w-none pt-3 text-slate-300
            prose-headings:text-slate-200 prose-headings:font-semibold
            prose-strong:text-slate-200 prose-code:text-sky-300
            prose-li:text-slate-300">
            <ReactMarkdown>{report}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  )
}
