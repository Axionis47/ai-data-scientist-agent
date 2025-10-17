import React from 'react'
import { Clock, Zap } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface Props {
  timings?: {
    stage_starts?: Record<string, number>
  }
  durations?: Record<string, number>
}

export function TimingDashboard({ timings, durations }: Props) {
  if (!timings && !durations) {
    return null
  }

  const stageStarts = timings?.stage_starts || {}
  const stages = Object.keys(stageStarts)

  // Calculate durations from stage starts
  const stageDurations: Record<string, number> = {}
  for (let i = 0; i < stages.length - 1; i++) {
    const current = stages[i]
    const next = stages[i + 1]
    stageDurations[current] = (stageStarts[next] - stageStarts[current]) / 1000 // Convert to seconds
  }

  // Add last stage duration if available
  if (durations) {
    Object.assign(stageDurations, durations)
  }

  const chartData = Object.entries(stageDurations).map(([stage, duration]) => ({
    stage: stage.toUpperCase(),
    duration: Number((duration / 1000).toFixed(2)), // Convert ms to seconds
  }))

  const totalTime = Object.values(stageDurations).reduce((sum, d) => sum + d, 0) / 1000

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <Clock size={20} />
        Performance Metrics
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="p-3 bg-surface rounded">
          <div className="flex items-center gap-2 mb-1">
            <Zap size={16} className="text-yellow-400" />
            <span className="text-sm text-muted">Total Time</span>
          </div>
          <div className="text-2xl font-bold">{totalTime.toFixed(2)}s</div>
        </div>

        {Object.entries(stageDurations).slice(0, 3).map(([stage, duration]) => (
          <div key={stage} className="p-3 bg-surface rounded">
            <div className="text-sm text-muted mb-1">{stage.toUpperCase()}</div>
            <div className="text-xl font-bold">{(duration / 1000).toFixed(2)}s</div>
          </div>
        ))}
      </div>

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="stage" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#111827',
                border: '1px solid #374151',
                borderRadius: '8px',
              }}
            />
            <Bar dataKey="duration" fill="#2563eb" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

