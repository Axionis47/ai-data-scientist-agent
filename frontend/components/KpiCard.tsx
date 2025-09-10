import React from 'react'
export function KpiCard({ label, value }: { label: string, value: string | number }){
  return (
    <div className="card kpi">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  )
}

