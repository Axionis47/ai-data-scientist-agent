import React from 'react'
import { AlertTriangle, AlertCircle, CheckCircle, Info } from 'lucide-react'

interface Issue {
  id: string
  severity: 'critical' | 'warning' | 'info'
  detail: string
}

interface Props {
  issues?: Issue[]
  recommendations?: string[]
}

export function DataQualityAlerts({ issues = [], recommendations = [] }: Props) {
  if (issues.length === 0 && recommendations.length === 0) {
    return (
      <div className="card bg-success-subtle">
        <div className="flex items-center gap-3">
          <CheckCircle size={24} className="text-success" />
          <div>
            <h4 className="m-0">Data Quality: Excellent</h4>
            <p className="text-muted m-0 mt-1">No critical issues detected</p>
          </div>
        </div>
      </div>
    )
  }

  const criticalIssues = issues.filter(i => i.severity === 'critical')
  const warnings = issues.filter(i => i.severity === 'warning')
  const infos = issues.filter(i => i.severity === 'info')

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <AlertTriangle size={20} />
        Data Quality Report
      </h3>

      {criticalIssues.length > 0 && (
        <div className="alert alert-error mb-3">
          <div className="flex items-start gap-2">
            <AlertCircle size={20} className="mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="m-0 mb-2">Critical Issues ({criticalIssues.length})</h4>
              <ul className="m-0 pl-4">
                {criticalIssues.map((issue, idx) => (
                  <li key={idx} className="mb-1">
                    <strong>{issue.id}:</strong> {issue.detail}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {warnings.length > 0 && (
        <div className="alert alert-warning mb-3">
          <div className="flex items-start gap-2">
            <AlertTriangle size={20} className="mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="m-0 mb-2">Warnings ({warnings.length})</h4>
              <ul className="m-0 pl-4">
                {warnings.map((issue, idx) => (
                  <li key={idx} className="mb-1">
                    <strong>{issue.id}:</strong> {issue.detail}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {infos.length > 0 && (
        <div className="alert alert-info mb-3">
          <div className="flex items-start gap-2">
            <Info size={20} className="mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="m-0 mb-2">Information ({infos.length})</h4>
              <ul className="m-0 pl-4">
                {infos.map((issue, idx) => (
                  <li key={idx} className="mb-1">
                    <strong>{issue.id}:</strong> {issue.detail}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {recommendations.length > 0 && (
        <div className="alert alert-info">
          <div className="flex items-start gap-2">
            <Info size={20} className="mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="m-0 mb-2">Recommendations</h4>
              <ul className="m-0 pl-4">
                {recommendations.map((rec, idx) => (
                  <li key={idx} className="mb-1">{rec}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

