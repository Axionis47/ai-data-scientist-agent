import React from 'react'
export function ProgressBar({ value }: { value: number }){
  return (
    <div className="progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value*100)} role="progressbar">
      <span style={{width: `${Math.min(100, Math.max(0, value*100))}%`}} />
    </div>
  )
}

