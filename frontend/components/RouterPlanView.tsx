import React from 'react'
import { GitBranch, CheckCircle, Circle } from 'lucide-react'

interface Props {
  routerPlan?: {
    decisions?: Record<string, any>
    plan?: string
    reasoning?: string
  }
}

export function RouterPlanView({ routerPlan }: Props) {
  if (!routerPlan) {
    return (
      <div className="card">
        <h3 className="flex items-center gap-2">
          <GitBranch size={20} />
          AI Router Plan
        </h3>
        <p className="text-muted">No router plan available</p>
      </div>
    )
  }

  const decisions = routerPlan.decisions || {}

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <GitBranch size={20} />
        AI Router Plan
      </h3>

      {routerPlan.reasoning && (
        <div className="alert alert-info mb-4">
          <h4 className="m-0 mb-2">Reasoning</h4>
          <p className="m-0">{routerPlan.reasoning}</p>
        </div>
      )}

      {routerPlan.plan && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold mb-2">Execution Plan</h4>
          <p className="text-sm">{routerPlan.plan}</p>
        </div>
      )}

      {Object.keys(decisions).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-3">Decisions Made</h4>
          <div className="space-y-2">
            {Object.entries(decisions).map(([key, value]) => (
              <div key={key} className="flex items-start gap-3 p-3 bg-surface rounded">
                <CheckCircle size={18} className="text-success mt-1 flex-shrink-0" />
                <div className="flex-1">
                  <div className="font-semibold text-sm">{formatKey(key)}</div>
                  <div className="text-sm text-muted mt-1">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function formatKey(key: string): string {
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

