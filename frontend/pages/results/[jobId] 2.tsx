import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { KpiCard } from '../../components/KpiCard'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Results(){
  const router = useRouter()
  const { jobId } = router.query
  const [result, setResult] = useState<any>(null)
  const [manifest, setManifest] = useState<any>(null)

  const [loading, setLoading] = useState(true)
  const [showModelingJson, setShowModelingJson] = useState(false)
  const [showFairnessJson, setShowFairnessJson] = useState(false)
  const [showReproJson, setShowReproJson] = useState(false)

  useEffect(()=>{
    if(!jobId) return
    const load = async () => {
      try{
        setLoading(true)
        const [r1, r2] = await Promise.all([
          fetch(`${API}/result/${jobId}`),
          fetch(`${API}/static/jobs/${jobId}/manifest.json`).catch(()=>null)
        ])
        const j1 = await r1.json(); setResult(j1)
        if(r2 && r2.ok){ try{ setManifest(await r2.json()) } catch(e){} }
      } finally {
        setLoading(false)
      }
    }
    load()
  },[jobId])

  const eda = result?.eda || {}
  const modeling = result?.modeling || {}
  const qa = result?.qa || {}
  const explain = result?.explain || {}

  return (
    <div>
      {loading && (
        <div className="card" style={{padding:'16px', marginBottom:16}}>
          <div style={{color:'#9ca3af'}}>Loading results…</div>
        </div>
      )}

    <div style={{padding:'24px 0'}}>
      <div className="card" style={{textAlign:'center', padding:'20px', marginBottom:16}}>
        <h1 style={{margin:0}}>Findings & Evidence</h1>
      </div>
      <div className="grid" style={{marginBottom:16}}>
        <div className="card" style={{flex:'1 1 260px'}}><KpiCard label="Columns" value={(eda.columns||[]).length} /></div>
        <div className="card" style={{flex:'1 1 260px'}}><KpiCard label="Task" value={modeling.task || 'N/A'} /></div>
        <div className="card" style={{flex:'1 1 260px'}}>
          {modeling.task === 'classification' ? (
            <KpiCard label="F1 (best)" value={typeof modeling?.best?.f1==='number'? modeling.best.f1.toFixed(3): 'N/A'} />
          ) : (
            <KpiCard label="R2 (best)" value={typeof modeling?.best?.r2==='number'? modeling.best.r2.toFixed(3): (typeof modeling?.best?.rmse==='number'? `RMSE ${modeling.best.rmse.toFixed(3)}` : 'N/A')} />
          )}
        </div>
      </div>

      <div className="card" style={{marginBottom:16}}>
        <h3>Executive Synopsis</h3>
        <p>Objective: {modeling.task || 'descriptive'}.</p>
        <p>Highlights: best={modeling?.best?.name || 'N/A'}; candidates={(modeling?.selected_tools||[]).join(', ') || 'n/a'}; features={modeling?.features? `${modeling.features.numeric||0} numeric, ${modeling.features.categorical||0} categorical` : 'n/a'}</p>
        {/* Applied decisions chips */}
        {(() => {
          const dec = (result?.router_plan?.decisions || {}) as any
          const chips = [] as string[]
          const profile = (manifest?.profile || '').toString().toLowerCase()
          if(profile) chips.push(`profile:${profile}`)
          if(dec.metric) chips.push(`metric:${dec.metric}`)
          if(dec.split) chips.push(`split:${dec.split}`)
          if(dec.budget) chips.push(`budget:${dec.budget}`)
          if(dec.class_weight) chips.push(`class_weight:${dec.class_weight}`)
          if(typeof dec.calibration !== 'undefined') chips.push(`calibration:${String(dec.calibration)}`)
          if(chips.length===0) return null
          return (
            <div style={{display:'flex', flexWrap:'wrap', gap:8, marginTop:8}}>
              {chips.map(c => {
                const key = (c.split(':')[0]||'').trim()
                const tips: Record<string,string> = {
                  profile: 'full: richer analysis; lean: faster, skips fairness and TS/text FE',
                  metric: 'Optimization objective (set by router/intent)',
                  split: 'CV strategy (time vs random)',
                  budget: 'Search/HPO effort budget',
                  class_weight: 'Handle imbalance by weighting classes',
                  calibration: 'Probability calibration for classifiers'
                }
                const title = tips[key] || c
                return (
                  <span key={c} title={title} style={{padding:'2px 8px', border:'1px solid #ddd', borderRadius:12, fontSize:12, background:'#fafafa'}}>{c}</span>
                )
              })}
            </div>
          )
        })()}
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

          )}
        </div>
      </div>

      <div className="grid" style={{marginBottom:16}}>
        <div className="card">
          <h3>Modeling Summary</h3>
          <div className="row" style={{justifyContent:'space-between', alignItems:'center'}}>
            <div style={{color:'#6b7280', fontSize:12}}>Best: {(modeling?.best?.name)||'N/A'}</div>
            <button className="btn secondary" onClick={()=>setShowModelingJson(v=>!v)}>{showModelingJson? 'Hide JSON':'Show JSON'}</button>
          </div>
          {showModelingJson && <pre>{JSON.stringify(modeling, null, 2)}</pre>}
        </div>
        <div className="card">
          <h3>Router Plan</h3>
          {result?.router_plan ? (
            <pre>{JSON.stringify(result.router_plan, null, 2)}</pre>
          ) : (
            <div>No router plan recorded.</div>
          )}
        </div>
      </div>

      <div className="grid" style={{marginBottom:16}}>
        <div className="card">
          <h3>Fairness (beta) <span style={{marginLeft:6, padding:'2px 6px', border:'1px solid #eee', borderRadius:8, fontSize:11}} title="See docs/COOKBOOK.md for details">beta</span></h3>
          {result?.fairness ? (
            <>
              {(() => {
                try {
                  const f = result.fairness || {}
                  const entries = Object.entries(f.summaries || {}) as [string, any][]
                  const ranked = entries.map(([col, val])=>({col, disparity: Number(val?.disparity||0)})).sort((a,b)=>b.disparity-a.disparity)
                  const top = ranked.slice(0,2)
                  if(top.length===0) return null
                  return (
                    <div style={{marginBottom:8, color:'#4b5563'}}>
                      <div style={{fontWeight:600, marginBottom:4}}>Top disparities</div>
                      {top.map(t => (
                        <div key={t.col}>
                          <span style={{fontFamily:'monospace'}}>{t.col}</span>: Δ {t.disparity.toFixed(3)}
                        </div>
                      ))}
                    </div>
                  )
                } catch(e) { return null }
              })()}
              <div className="row" style={{justifyContent:'flex-end'}}>
                <button className="btn secondary" onClick={()=>setShowFairnessJson(v=>!v)}>{showFairnessJson? 'Hide JSON':'Show JSON'}</button>
              </div>
              {showFairnessJson && <pre>{JSON.stringify(result.fairness, null, 2)}</pre>}
            </>
          ) : (
            <div>No slice metrics computed.</div>
          )}
        </div>
        <div className="card">
          <h3>Reproducibility</h3>
          {result?.reproducibility ? (
            <>
              <div className="row" style={{justifyContent:'flex-end'}}>
                <button className="btn secondary" onClick={()=>setShowReproJson(v=>!v)}>{showReproJson? 'Hide JSON':'Show JSON'}</button>
              </div>
              {showReproJson && <pre>{JSON.stringify(result.reproducibility, null, 2)}</pre>}
            </>
          ) : (
            <div>No reproducibility record.</div>
          )}
        </div>
      </div>

      <div className="grid" style={{marginBottom:16}}>
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
    </div>
  )
}

