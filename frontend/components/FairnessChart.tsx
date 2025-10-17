import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Scale } from 'lucide-react'

interface SliceMetric {
  column: string
  groups: Record<string, any>
  disparity?: number
}

interface Props {
  fairness?: {
    columns?: string[]
    summaries?: Record<string, SliceMetric>
    notes?: string[]
  }
}

export function FairnessChart({ fairness }: Props) {
  if (!fairness || !fairness.summaries || Object.keys(fairness.summaries).length === 0) {
    return (
      <div className="card">
        <h3 className="flex items-center gap-2">
          <Scale size={20} />
          Fairness Analysis
        </h3>
        <p className="text-muted">No fairness metrics available</p>
      </div>
    )
  }

  const summaries = fairness.summaries

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <Scale size={20} />
        Fairness Analysis
      </h3>

      {Object.entries(summaries).map(([column, data]) => {
        const groups = data.groups || {}
        const chartData = Object.entries(groups).map(([group, metrics]: [string, any]) => ({
          group,
          prevalence: metrics.prevalence || 0,
          accuracy: metrics.accuracy || 0,
          f1: metrics.f1 || 0,
          support: metrics.support || 0,
        }))

        const disparity = data.disparity || 0

        return (
          <div key={column} className="mb-6">
            <h4 className="mb-2">
              {column}
              {disparity > 0.1 && (
                <span className="badge badge-warning ml-2">
                  Disparity: {(disparity * 100).toFixed(1)}%
                </span>
              )}
            </h4>

            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="group" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Bar dataKey="prevalence" fill="#60a5fa" name="Prevalence" />
                <Bar dataKey="accuracy" fill="#34d399" name="Accuracy" />
                <Bar dataKey="f1" fill="#a78bfa" name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-3 grid grid-cols-2 gap-2">
              {Object.entries(groups).map(([group, metrics]: [string, any]) => (
                <div key={group} className="p-2 bg-surface rounded">
                  <div className="font-semibold">{group}</div>
                  <div className="text-sm text-muted">
                    Support: {metrics.support || 0} | 
                    Prevalence: {((metrics.prevalence || 0) * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })}

      {fairness.notes && fairness.notes.length > 0 && (
        <div className="alert alert-info mt-4">
          <h5 className="m-0 mb-2">Notes</h5>
          <ul className="m-0 pl-4">
            {fairness.notes.map((note, idx) => (
              <li key={idx}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

