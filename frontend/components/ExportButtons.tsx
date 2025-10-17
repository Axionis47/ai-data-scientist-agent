import React from 'react'
import { Download, FileText, FileJson, Image, Database } from 'lucide-react'

interface Props {
  jobId: string
  apiUrl: string
  result?: any
}

export function ExportButtons({ jobId, apiUrl, result }: Props) {
  const downloadFile = async (url: string, filename: string) => {
    try {
      const response = await fetch(url)
      const blob = await response.blob()
      const link = document.createElement('a')
      link.href = window.URL.createObjectURL(blob)
      link.download = filename
      link.click()
    } catch (error) {
      console.error('Download failed:', error)
      alert('Download failed. Please try again.')
    }
  }

  const downloadJSON = (data: any, filename: string) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const link = document.createElement('a')
    link.href = window.URL.createObjectURL(blob)
    link.download = filename
    link.click()
  }

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <Download size={20} />
        Export Results
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <button
          className="btn btn-outline flex items-center justify-center gap-2"
          onClick={() => downloadFile(`${apiUrl}/static/jobs/${jobId}/report.html`, `report-${jobId}.html`)}
        >
          <FileText size={18} />
          Full Report (HTML)
        </button>

        <button
          className="btn btn-outline flex items-center justify-center gap-2"
          onClick={() => result && downloadJSON(result, `results-${jobId}.json`)}
          disabled={!result}
        >
          <FileJson size={18} />
          Results (JSON)
        </button>

        <button
          className="btn btn-outline flex items-center justify-center gap-2"
          onClick={() => downloadFile(`${apiUrl}/static/jobs/${jobId}/manifest.json`, `manifest-${jobId}.json`)}
        >
          <Database size={18} />
          Manifest
        </button>

        <button
          className="btn btn-outline flex items-center justify-center gap-2"
          onClick={() => {
            // Download all plots as a zip (simplified - just open in new tab for now)
            window.open(`${apiUrl}/static/jobs/${jobId}/plots/`, '_blank')
          }}
        >
          <Image size={18} />
          Plots & Charts
        </button>
      </div>

      <div className="mt-3 text-sm text-muted">
        💡 Tip: All artifacts are also available at <code>/static/jobs/{jobId}/</code>
      </div>
    </div>
  )
}

