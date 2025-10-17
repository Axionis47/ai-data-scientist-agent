import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { Play, Pause, CheckCircle, XCircle, Clock, MessageSquare } from 'lucide-react'
import { AgentLog } from '../../components/AgentLog'
import { ProgressBar } from '../../components/ProgressBar'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type Status = { job_id: string, status: string, progress: number, stage?: string, messages: {role:string,content:string}[] }

const stageInfo: Record<string, { label: string, color: string, description: string }> = {
  ingest: { label: 'Ingesting Data', color: '#60a5fa', description: 'Loading and validating your dataset' },
  eda: { label: 'Exploratory Analysis', color: '#60a5fa', description: 'Analyzing patterns, distributions, and correlations' },
  modeling: { label: 'Training Models', color: '#a78bfa', description: 'Training multiple ML models and selecting the best' },
  report: { label: 'Generating Report', color: '#fb923c', description: 'Creating comprehensive analysis report' },
  qa: { label: 'Quality Assurance', color: '#2dd4bf', description: 'Validating results and checking for issues' }
}

export default function Run(){
  const router = useRouter()
  const { jobId } = router.query
  const [status, setStatus] = useState<Status|null>(null)
  const [result, setResult] = useState<any>(null)

  useEffect(()=>{
    if(!jobId) return
    const int = setInterval(async ()=>{
      const r = await fetch(`${API}/status/${jobId}`)
      const j = await r.json(); setStatus(j)
      // fetch result early during EDA for preview
      if(j.stage === 'eda' || j.status === 'COMPLETED'){
        try{ const rr = await fetch(`${API}/result/${jobId}`); if(rr.ok){ setResult(await rr.json()) } }catch{}
      }
      if(j.status === 'COMPLETED'){
        clearInterval(int)
      }
    }, 1000)
    return ()=>clearInterval(int)
  },[jobId])

  const sendClarification = async () => {
    if(!jobId) return
    const msg = prompt('Clarification (e.g., target=Survived)')
    if(!msg) return
    await fetch(`${API}/clarify`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({job_id: jobId, message: msg})})
  }

  const currentStage = status?.stage || 'starting'
  const stageData = stageInfo[currentStage] || { label: 'Starting', color: '#60a5fa', description: 'Initializing analysis' }
  const isComplete = status?.status === 'COMPLETED'
  const isFailed = status?.status === 'FAILED'

  return (
    <div style={{padding:'24px 0'}}>
      {/* Header */}
      <div className="card" style={{textAlign:'center', padding:'32px', marginBottom:24}}>
        <div className="flex items-center justify-center gap-3 mb-3">
          {isComplete ? (
            <CheckCircle size={40} className="text-success" />
          ) : isFailed ? (
            <XCircle size={40} style={{color:'#ef4444'}} />
          ) : (
            <Play size={40} style={{color:stageData.color}} />
          )}
          <h1 style={{margin:0, fontSize:36}}>
            {isComplete ? 'Analysis Complete!' : isFailed ? 'Analysis Failed' : 'Analysis In Progress'}
          </h1>
        </div>
        <p className="text-muted">{isComplete ? 'Your results are ready' : stageData.description}</p>
      </div>

      {/* Progress Card */}
      <div className="card" style={{marginBottom:24}}>
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Clock size={18} className="text-muted" />
              <span className="font-semibold">Current Stage: {stageData.label}</span>
            </div>
            <span className="badge" style={{backgroundColor:stageData.color, borderColor:stageData.color}}>
              {Math.round(status?.progress ?? 0)}%
            </span>
          </div>
          <ProgressBar value={status?.progress ?? 0} />
        </div>

        {/* Stage Timeline */}
        <div className="flex items-center justify-between mt-4" style={{position:'relative'}}>
          {Object.entries(stageInfo).map(([key, info], idx) => {
            const isActive = key === currentStage
            const isPast = ['ingest', 'eda', 'modeling', 'report', 'qa'].indexOf(key) < ['ingest', 'eda', 'modeling', 'report', 'qa'].indexOf(currentStage)
            return (
              <div key={key} className="flex flex-col items-center" style={{flex:1, position:'relative'}}>
                <div
                  style={{
                    width:32,
                    height:32,
                    borderRadius:'50%',
                    background: isActive ? info.color : isPast ? '#10b981' : '#1f2937',
                    border: `2px solid ${isActive ? info.color : isPast ? '#10b981' : '#374151'}`,
                    display:'flex',
                    alignItems:'center',
                    justifyContent:'center',
                    marginBottom:8,
                    transition:'all 0.3s ease'
                  }}
                >
                  {isPast && <CheckCircle size={18} style={{color:'white'}} />}
                </div>
                <div className="text-xs text-center" style={{color: isActive ? info.color : '#9ca3af'}}>
                  {info.label.split(' ')[0]}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 justify-center mb-6">
        <button
          className="btn"
          onClick={()=>router.push(`/results/${jobId}`)}
          disabled={!result}
        >
          View Results
        </button>
        <button
          className="btn secondary"
          onClick={sendClarification}
          disabled={!jobId || isComplete}
        >
          <MessageSquare size={18} className="inline mr-2" />
          Send Clarification
        </button>
        <button
          className="btn btn-outline"
          onClick={async()=>{
            if(jobId){
              await fetch(`${API}/cancel/${jobId}`, {method:'POST'})
              alert('Analysis cancelled')
              router.push('/')
            }
          }}
          disabled={!jobId || isComplete}
          style={{borderColor:'#ef4444', color:'#ef4444'}}
        >
          Cancel
        </button>
      </div>

      {/* Agent Log and Intermediate Results */}
      <div className="grid" style={{gap:24}}>
        <div className="card" style={{flex:'1 1 500px'}}>
          <h3 className="flex items-center gap-2 mb-4">
            <MessageSquare size={20} />
            Agent Reasoning Stream
          </h3>
          <AgentLog messages={status?.messages || []} />
        </div>

        {result?.eda && (
          <div className="card" style={{flex:'1 1 400px'}}>
            <h3 className="mb-4">Early Insights</h3>
            <div className="space-y-3">
              <div className="p-3 bg-surface rounded">
                <div className="text-sm text-muted mb-1">Columns Detected</div>
                <div className="text-2xl font-bold">{(result.eda.columns||[]).length}</div>
              </div>
              {result.eda.summary && (
                <div className="p-3 bg-surface rounded">
                  <div className="text-sm text-muted mb-1">Quick Summary</div>
                  <div className="text-sm">{result.eda.summary}</div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

