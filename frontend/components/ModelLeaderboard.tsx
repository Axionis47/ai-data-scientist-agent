import React from 'react'
import { TrendingUp, Award, Clock } from 'lucide-react'

interface ModelCandidate {
  name: string
  f1?: number
  accuracy?: number
  r2?: number
  rmse?: number
  cv_mean?: number
  cv_std?: number
  train_time_s?: number
  is_best?: boolean
}

interface Props {
  candidates: ModelCandidate[]
  task: 'classification' | 'regression'
}

export function ModelLeaderboard({ candidates, task }: Props) {
  if (!candidates || candidates.length === 0) {
    return (
      <div className="card">
        <h3 className="flex items-center gap-2">
          <TrendingUp size={20} />
          Model Leaderboard
        </h3>
        <p className="text-muted">No models trained yet</p>
      </div>
    )
  }

  const sortedCandidates = [...candidates].sort((a, b) => {
    if (task === 'classification') {
      return (b.f1 || 0) - (a.f1 || 0)
    }
    return (b.r2 || 0) - (a.r2 || 0)
  })

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <TrendingUp size={20} />
        Model Leaderboard
      </h3>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              {task === 'classification' ? (
                <>
                  <th>F1 Score</th>
                  <th>Accuracy</th>
                </>
              ) : (
                <>
                  <th>R² Score</th>
                  <th>RMSE</th>
                </>
              )}
              <th>CV Mean</th>
              <th>CV Std</th>
              <th>Time (s)</th>
            </tr>
          </thead>
          <tbody>
            {sortedCandidates.map((model, idx) => (
              <tr key={idx} className={model.is_best ? 'highlight-row' : ''}>
                <td>
                  {idx === 0 && <Award size={16} className="inline text-yellow-400 mr-1" />}
                  {idx + 1}
                </td>
                <td className="font-semibold">
                  {model.name}
                  {model.is_best && <span className="badge ml-2">Best</span>}
                </td>
                {task === 'classification' ? (
                  <>
                    <td>{model.f1?.toFixed(4) || 'N/A'}</td>
                    <td>{model.accuracy?.toFixed(4) || 'N/A'}</td>
                  </>
                ) : (
                  <>
                    <td>{model.r2?.toFixed(4) || 'N/A'}</td>
                    <td>{model.rmse?.toFixed(4) || 'N/A'}</td>
                  </>
                )}
                <td>{model.cv_mean?.toFixed(4) || 'N/A'}</td>
                <td className="text-muted">{model.cv_std?.toFixed(4) || 'N/A'}</td>
                <td className="flex items-center gap-1">
                  <Clock size={14} className="text-muted" />
                  {model.train_time_s?.toFixed(2) || 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

