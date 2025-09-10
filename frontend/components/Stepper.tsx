import React from 'react'

type Step = { key: string; label: string }
export function Stepper({ steps, active }: { steps: Step[]; active: string }) {
  return (
    <div className="stepper" role="tablist" aria-label="Setup steps">
      {steps.map(s => (
        <div key={s.key} className={`step ${s.key===active?'active':''}`} role="tab" aria-selected={s.key===active} aria-controls={`panel-${s.key}`}>
          <span className="badge">{s.label}</span>
        </div>
      ))}
    </div>
  )
}

