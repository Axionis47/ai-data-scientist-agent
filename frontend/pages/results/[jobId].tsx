import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { KpiCard } from '../../components/KpiCard'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Results(){
  const router = useRouter()
  const { jobId } = router.query
  const [result, setResult] = useState<any>(null)

  useEffect(()=>{
    if(!jobId) return
    const load = async () => {
      const r = await fetch(`${API}/result/${jobId}`)
      const j = await r.json(); setResult(j)
    }
    load()
  },[jobId])

  const eda = result?.eda || {}
  const modeling = result?.modeling || {}
  const qa = result?.qa || {}
  const explain = result?.explain || {}

  return (
    <div style={{padding:'24px 0'}}>
      <div className="card" style={{textAlign:'center', padding:'20px', marginBottom:16}}>
        <h1 style={{margin:0}}>Findings & Evidence</h1>
      </div>
      <div className="grid" style={{marginBottom:16}}>
        <div className="card" style={{flex:'1 1 260px'}}><KpiCard label="Columns" value={(eda.columns||[]).length} /></div>
        <div className="card" style={{flex:'1 1 260px'}}><KpiCard label="Task" value={modeling.task || 'N/A'} /></div>
        <div className="card" style={{flex:'1 1 260px'}}><KpiCard label="Score" value={typeof modeling.score==='number'? modeling.score.toFixed(3): 'N/A'} /></div>
      </div>

      <div className="card" style={{marginBottom:16}}>
        <h3>Executive Synopsis</h3>
        <p>Objective: {modeling.task || 'descriptive'}. Target: {modeling.target || 'N/A'}.</p>
        <p>Highlights: {modeling.note || 'Baseline built when a target is provided; EDA always available.'}</p>
      </div>

      <div className="grid" style={{marginBottom:16}}>
        <div className="card">
          <h3>EDA Summary</h3>
          <pre>{JSON.stringify(eda, null, 2)}</pre>
        </div>
        <div className="card">
          <h3>EDA Plots</h3>
          {eda?.plots ? (
            <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(220px,1fr))', gap:12}}>
              {eda.plots.missingness && (
                <div>
                  <div style={{fontWeight:600, marginBottom:4}}>Missingness</div>
                  <img src={`${API}${eda.plots.missingness}`} style={{width:'100%'}}/>
                </div>
              )}
              {(eda.plots.histograms||[]).map((p:string)=> (
                <div key={p}>
                  <div style={{fontWeight:600, marginBottom:4}}>Histogram</div>
                  <img src={`${API}${p}`} style={{width:'100%'}}/>
                </div>
              ))}
              {(eda.plots.categoricals||[]).map((p:string)=> (
                <div key={p}>
                  <div style={{fontWeight:600, marginBottom:4}}>Categorical</div>
                  <img src={`${API}${p}`} style={{width:'100%'}}/>
                </div>
              ))}
            </div>
          ) : (
            <div>No plots generated.</div>
      {modeling?.card && (
        <div className="card" style={{marginBottom:16}}>
          <h3>Model Card</h3>
          <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))', gap:12}}>
            <div>
              <div style={{fontWeight:600}}>Model</div>
              <div>{modeling.card.model_name}</div>
            </div>
            <div>
              <div style={{fontWeight:600}}>Primary Metric</div>
              <div>{modeling.card.primary_metric_name}: {modeling.card.primary_metric_value}</div>
            </div>
            {modeling.card?.threshold && (
              <div>
                <div style={{fontWeight:600}}>Threshold</div>
                <div>{modeling.card.threshold}</div>
              </div>
            )}
            <div>
              <div style={{fontWeight:600}}>CV Strategy</div>
              <div>{modeling.card.cv_strategy}</div>
            </div>
          </div>
          <div style={{marginTop:12}}>
            <div style={{fontWeight:600}}>Top Features</div>
            <pre style={{whiteSpace:'pre-wrap'}}>{JSON.stringify(modeling.card.top_features, null, 2)}</pre>
          </div>
          {modeling.card?.warnings?.length ? (
            <div style={{marginTop:12}}>
              <div style={{fontWeight:600}}>Warnings</div>
              <ul>
                {modeling.card.warnings.map((w:string, i:number)=> (<li key={i}>{w}</li>))}
              </ul>
            </div>
          ) : null}
        </div>
      )}

          )}
        </div>
      </div>

      <div className="grid" style={{marginBottom:16}}>
        <div className="card">
          <h3>Modeling Summary</h3>
          <pre>{JSON.stringify(modeling, null, 2)}</pre>
        </div>
        <div className="card">
          <h3>Explainability</h3>
          {explain?.pdp || explain?.roc || explain?.pr ? (
            <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(220px,1fr))', gap:12}}>
              {explain.roc && (
                <div>
                  <div style={{fontWeight:600, marginBottom:4}}>ROC Curve</div>
                  <img src={`${API}${explain.roc}`} style={{width:'100%'}}/>
                </div>
              )}
              {explain.pr && (
                <div>
                  <div style={{fontWeight:600, marginBottom:4}}>PR Curve</div>
                  <img src={`${API}${explain.pr}`} style={{width:'100%'}}/>
                </div>
              )}
              {(explain.pdp||[]).map((p:string)=> (
                <div key={p}>
                  <div style={{fontWeight:600, marginBottom:4}}>PDP</div>
                  <img src={`${API}${p}`} style={{width:'100%'}}/>
                </div>
              ))}
            </div>
          ) : (
            <pre>{JSON.stringify(explain, null, 2)}</pre>
          )}
        </div>
      </div>

      <div className="card" style={{marginBottom:16}}>
        <h3>QA Findings</h3>
        <pre>{JSON.stringify(qa, null, 2)}</pre>
      </div>

      {result?.report_html && (
        <div className="card">
          <h3>Full Report</h3>
          <iframe srcDoc={result.report_html} style={{width:'100%', height:500}} />
        </div>
      )}
    </div>
  )
}

