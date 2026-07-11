import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, Loader2 } from 'lucide-react'

interface Props {
  onResult: (result: unknown) => void
  onError: (msg: string) => void
}

export default function ImageUploader({ onResult, onError }: Props) {
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [filename, setFilename] = useState('')

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (!file) return

      setFilename(file.name)
      setPreview(URL.createObjectURL(file))
      setLoading(true)
      onError('')

      const form = new FormData()
      form.append('file', file)

      try {
        const res = await fetch('/api/analyze', { method: 'POST', body: form })
        if (!res.ok) {
          const err = await res.json()
          throw new Error(err.detail || 'Analysis failed')
        }
        const data = await res.json()
        onResult(data)
      } catch (e) {
        onError(e instanceof Error ? e.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    },
    [onResult, onError],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    maxFiles: 1,
    disabled: loading,
  })

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
          transition-all duration-200
          ${isDragActive ? 'border-sky-400 bg-sky-950/40' : 'border-slate-700 hover:border-slate-500 bg-slate-900/40'}
          ${loading ? 'opacity-60 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        {preview ? (
          <div className="space-y-3">
            <img
              src={preview}
              alt="Uploaded"
              className="mx-auto max-h-56 rounded-xl object-contain"
            />
            <p className="text-sm text-slate-400 truncate">{filename}</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="mx-auto w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center">
              <ImageIcon className="w-7 h-7 text-slate-400" />
            </div>
            <div>
              <p className="text-slate-300 font-medium">Drop an image here</p>
              <p className="text-sm text-slate-500 mt-1">or click to browse — JPG, PNG, WebP up to 10MB</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center rounded-2xl bg-slate-950/80 gap-3">
            <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
            <p className="text-sm text-sky-300 font-medium">Running analysis pipeline…</p>
            <p className="text-xs text-slate-500">ML detector · frequency analysis · metadata · AI agent</p>
          </div>
        )}
      </div>

      {!loading && preview && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            setPreview(null)
            setFilename('')
            onResult(null)
          }}
          className="w-full py-2 rounded-xl border border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-500 text-sm transition-colors"
        >
          <Upload className="inline w-4 h-4 mr-2" />
          Upload a different image
        </button>
      )}
    </div>
  )
}
