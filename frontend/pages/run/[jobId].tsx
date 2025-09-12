import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { AgentLog } from '../../components/AgentLog'
import { ProgressBar } from '../../components/ProgressBar'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type Status = { job_id: string, status: string, progress: number, stage?: string, messages: {role:string,content:string}[] }

enum StageColor { eda='#60a5fa', modeling='#a78bfa', report='#fb923c', qa='#2dd4bf' }

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

  return (
    <div style={{padding:'24px 0'}}>
      <div className="card" style={{textAlign:'center', padding:'20px', marginBottom:16}}>
        <h1 style={{margin:0}}>Analysis In Motion</h1>
      </div>
      <div className="card" style={{marginBottom:16}}>
        <div className="row" style={{alignItems:'center'}}>
          <div className="col"><div className="badge">Stage: {status?.stage || 'starting'}</div></div>
          <div className="col"><ProgressBar value={status?.progress ?? 0} /></div>
          <div className="col" style={{textAlign:'right'}}>
            <button className="btn secondary" onClick={()=>router.push(`/results/${jobId}`)} disabled={!result}>View Results</button>
            <button className="btn" onClick={sendClarification} disabled={!jobId}>Send Clarification</button>
            <button className="btn danger" onClick={async()=>{ if(jobId){ await fetch(`${API}/cancel/${jobId}`, {method:'POST'}); alert('Cancelled'); } }} disabled={!jobId}>Cancel</button>
          </div>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h3>Agent Reasoning Stream</h3>
          <AgentLog messages={status?.messages || []} />
        </div>
        <div className="card">
          <h3>Intermediate</h3>
          <div className="skeleton" style={{height:160, borderRadius:8}} />
        </div>
      </div>
    </div>
  )
}

