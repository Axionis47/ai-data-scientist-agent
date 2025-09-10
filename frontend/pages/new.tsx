import { useEffect, useMemo, useState } from 'react'
import { Stepper } from '../components/Stepper'
import { FileDropZone } from '../components/FileDropZone'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type Uploaded = { job_id: string, dataset_path: string, file_format: string }

export default function NewAnalysis(){
  const steps = [
    {key:'data', label:'Data'},
    {key:'context', label:'Context'},
    {key:'question', label:'Question'},
    {key:'review', label:'Review'},
  ]
  const [active, setActive] = useState('data')
  const [uploaded, setUploaded] = useState<Uploaded|null>(null)
  const [datasetPath, setDatasetPath] = useState('')
  const [context, setContext] = useState('Business: Transportation. Dataset: Titanic passengers (1912), demographics and tickets.')
  const [question, setQuestion] = useState('Classify which passengers survived. target=Survived')
  const [fileFormat, setFileFormat] = useState<string|undefined>(undefined)
  const [plan, setPlan] = useState<string>('')
  
  const canNext = useMemo(()=>{
    if(active==='data') return Boolean(uploaded?.dataset_path || datasetPath)
    if(active==='context') return context.length>10
    if(active==='question') return question.length>10
    return true
  },[active, uploaded, datasetPath, context, question])

  const onFile = async (f: File) => {
    const fd = new FormData(); fd.append('file', f)
    const r = await fetch(`${API}/upload`, {method:'POST', body: fd})
    const j = await r.json(); setUploaded(j); setDatasetPath(j.dataset_path); setFileFormat(j.file_format)
  }

  useEffect(()=>{
    if(active==='review'){
      const ds = datasetPath || uploaded?.dataset_path
      setPlan(`Plan:\n- Load ${ds}\n- EDA: profile dtypes, missingness, correlations\n- Modeling: baseline model if target present\n- Report + QA\n`)
    }
  },[active, datasetPath, uploaded])

  const start = async () => {
    const body = { dataset_path: datasetPath || uploaded?.dataset_path, nl_description: context, question, file_format: fileFormat }
    const r = await fetch(`${API}/analyze`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
    const j = await r.json();
    const jobs = JSON.parse(localStorage.getItem('jobs')||'[]'); jobs.unshift(j.job_id); localStorage.setItem('jobs', JSON.stringify(jobs.slice(0,20)))
    window.location.href = `/run/${j.job_id}`
  }

  return (
    <div style={{padding:'24px 0'}}>
      <div style={{display:'flex', justifyContent:'center', marginBottom:12}}>
        <Stepper steps={steps} active={active} />
      </div>

      {active==='data' && (
        <section id="panel-data" style={{marginTop:16}}>
          <div className="grid">
            <div className="card" style={{flex:'1 1 420px'}}>
              <h3>Upload Data</h3>
              <p style={{color:'#9ca3af',marginTop:4}}>Drag & drop your file. We’ll auto-profile columns, types, and missingness.</p>
              <FileDropZone onFile={onFile} />
            </div>
            <div className="card" style={{flex:'1 1 420px'}}>
              <h3>Or specify path</h3>
              <label className="label">Dataset path (local or mounted)</label>
              <input className="input" placeholder="backend/data/sample/titanic_small.csv" value={datasetPath} onChange={e=>setDatasetPath(e.target.value)} />
              <div style={{marginTop:8, color:'#9ca3af'}}>Supported: CSV, JSON, Parquet</div>
            </div>
          </div>
        </section>
      )}

      {active==='context' && (
        <section id="panel-context" style={{marginTop:16}}>
          <div className="card">
            <h3>Context & Assumptions</h3>
            <label className="label">Business goal, data provenance, definitions, time windows, known caveats. The agent uses this to guide feature logic and guard against spurious conclusions.</label>
            <textarea className="input" rows={8} value={context} onChange={e=>setContext(e.target.value)} />
          </div>
        </section>
      )}

      {active==='question' && (
        <section id="panel-question" style={{marginTop:16}}>
          <div className="grid">
            <div className="card">
              <h3>Core Question</h3>
              <input className="input" value={question} onChange={e=>setQuestion(e.target.value)} />
              <div style={{marginTop:8, color:'#9ca3af'}}>Tip: add target=&lt;column&gt; to train a model. Examples below.</div>
            </div>
            <div className="card">
              <h3>Templates</h3>
              <p style={{color:'#9ca3af',marginTop:4}}>Choose a pattern. We’ll align metrics and defaults automatically.</p>
              <div className="row">
                <button className="btn secondary" onClick={()=>setQuestion('Classify which passengers survived. target=Survived')}>Classification</button>
                <button className="btn secondary" onClick={()=>setQuestion('Predict fare amount based on passenger attributes. target=Fare')}>Regression</button>
                <button className="btn secondary" onClick={()=>setQuestion('What are the key drivers of survival?')}>Descriptive</button>
              </div>
            </div>
          </div>
        </section>
      )}

      {active==='review' && (
        <section id="panel-review" style={{marginTop:16}}>
          <div className="grid">
            <div className="card"><h3>Review Plan</h3><p style={{color:'#9ca3af'}}>A concise blueprint derived from your inputs.</p><pre>{plan}</pre></div>
            <div className="card"><h3>Inputs</h3><pre>{JSON.stringify({dataset_path: datasetPath || uploaded?.dataset_path, context, question}, null, 2)}</pre></div>
          </div>
        </section>
      )}

      <div className="row" style={{marginTop:16, justifyContent:'center', gap:12}}>
        <button className="btn secondary" onClick={()=>{
          const idx = steps.findIndex(s=>s.key===active); if(idx>0) setActive(steps[idx-1].key)
        }}>Back</button>
        {active!=='review' && <button className="btn" disabled={!canNext} onClick={()=>{
          const idx = steps.findIndex(s=>s.key===active); if(idx<steps.length-1) setActive(steps[idx+1].key)
        }}>Next</button>}
        {active==='review' && <button className="btn" onClick={start}>Start Analysis</button>}
      </div>
    </div>
  )
}

