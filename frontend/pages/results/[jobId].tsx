import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { 
  BarChart3, 
  Brain, 
  Scale, 
  Download, 
  FileText, 
  GitBranch
} from 'lucide-react'
import { KpiCard } from '../../components/KpiCard'
import { ModelLeaderboard } from '../../components/ModelLeaderboard'
import { DataQualityAlerts } from '../../components/DataQualityAlerts'
import { FairnessChart } from '../../components/FairnessChart'
import { FeatureEngineeringView } from '../../components/FeatureEngineeringView'
import { TimingDashboard } from '../../components/TimingDashboard'
import { ExportButtons } from '../../components/ExportButtons'
import { RouterPlanView } from '../../components/RouterPlanView'
import { Tabs } from '../../components/Tabs'
import { CollapsibleSection } from '../../components/CollapsibleSection'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Results(){
  const router = useRouter()
  const { jobId } = router.query
  const [result, setResult] = useState<any>(null)
  const [manifest, setManifest] = useState<any>(null)
  const [loading, setLoading] = useState(true)

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
  const fairness = result?.fairness || {}
  const reproducibility = result?.reproducibility || {}
  const routerPlan = result?.router_plan || manifest?.router_plan || {}
  const critique = result?.critique || {}

  const candidates = modeling?.candidates || []
  const task = modeling?.task || 'classification'

  if (loading) {
    return (
      <div style={{padding:'24px 0'}}>
        <div className="card" style={{textAlign:'center', padding:'40px'}}>
          <div className="skeleton" style={{height:200, borderRadius:12}} />
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div style={{padding:'24px 0'}}>
        <div className="card" style={{textAlign:'center', padding:'40px'}}>
          <h2>Results not found</h2>
          <p className="text-muted">Job ID: {jobId}</p>
        </div>
      </div>
    )
  }

  return (
    <div style={{padding:'24px 0'}}>
      <div className="card" style={{textAlign:'center', padding:'32px', marginBottom:24}}>
        <h1 style={{margin:0, fontSize:36}}>Analysis Complete</h1>
        <p className="text-muted" style={{marginTop:8}}>Job ID: {jobId}</p>
      </div>

      <div className="grid-cols-4" style={{marginBottom:24}}>
        <div className="card"><KpiCard label="Columns" value={(eda.columns||[]).length} /></div>
        <div className="card"><KpiCard label="Task Type" value={modeling.task || 'N/A'} /></div>
        <div className="card">
          {modeling.task === 'classification' ? (
            <KpiCard label="F1 Score (Best)" value={typeof modeling?.best?.f1==='number'? modeling.best.f1.toFixed(4): 'N/A'} />
          ) : (
            <KpiCard label="R² Score (Best)" value={typeof modeling?.best?.r2==='number'? modeling.best.r2.toFixed(4): 'N/A'} />
          )}
        </div>
        <div className="card"><KpiCard label="Models Trained" value={candidates.length || 1} /></div>
      </div>

      {critique?.issues && critique.issues.length > 0 && (
        <div style={{marginBottom:24}}>
          <DataQualityAlerts issues={critique.issues} recommendations={critique.recommendations} />
        </div>
      )}

      <Tabs defaultTab="overview" tabs={[
        {
          id: 'overview',
          label: 'Overview',
          icon: <BarChart3 size={18} />,
          content: (
            <div className="space-y-4">
              <ModelLeaderboard candidates={candidates} task={task} />
              <FeatureEngineeringView eda={eda} modeling={modeling} />
              <TimingDashboard timings={result?.timings} durations={result?.durations_ms} />
            </div>
          )
        },
        {
          id: 'eda',
          label: 'Data Exploration',
          icon: <BarChart3 size={18} />,
          content: <EdaSection eda={eda} jobId={jobId as string} apiUrl={API} />
        },
        {
          id: 'modeling',
          label: 'Modeling',
          icon: <Brain size={18} />,
          content: <ModelingSection modeling={modeling} explain={explain} jobId={jobId as string} apiUrl={API} />
        },
        {
          id: 'fairness',
          label: 'Fairness',
          icon: <Scale size={18} />,
          content: <FairnessChart fairness={fairness} />
        },
        {
          id: 'insights',
          label: 'AI Insights',
          icon: <GitBranch size={18} />,
          content: (
            <div className="space-y-4">
              <RouterPlanView routerPlan={routerPlan} />
              {qa?.findings && qa.findings.length > 0 && (
                <div className="card">
                  <h3>QA Findings</h3>
                  <ul>{qa.findings.map((f: any, i: number) => (<li key={i}>{f.message || f}</li>))}</ul>
                </div>
              )}
              {reproducibility && (
                <CollapsibleSection title="Reproducibility" defaultOpen={false}>
                  <div className="card"><pre style={{fontSize:12, overflow:'auto'}}>{JSON.stringify(reproducibility, null, 2)}</pre></div>
                </CollapsibleSection>
              )}
            </div>
          )
        },
        {
          id: 'report',
          label: 'Full Report',
          icon: <FileText size={18} />,
          content: result?.report_html ? (
            <iframe srcDoc={result.report_html} style={{width:'100%', height:'800px', border:'1px solid #1f2937', borderRadius:'12px', background:'white'}} />
          ) : (
            <div className="card"><p className="text-muted">No HTML report available</p></div>
          )
        },
        {
          id: 'export',
          label: 'Export',
          icon: <Download size={18} />,
          content: <ExportButtons jobId={jobId as string} apiUrl={API} result={result} />
        }
      ]} />

      <div style={{textAlign:'center', marginTop:40}}>
        <button className="btn secondary" onClick={() => router.push('/')}>Back to Home</button>
      </div>
    </div>
  )
}

function EdaSection({ eda, jobId, apiUrl }: { eda: any, jobId: string, apiUrl: string }) {
  const columns = eda?.columns || []
  const numericCols = columns.filter((c: any) => c.type === 'numeric')
  const categoricalCols = columns.filter((c: any) => c.type === 'categorical')

  return (
    <>
      <div className="card">
        <h3 className="mb-4">Dataset Summary</h3>
        <div className="grid-cols-3 mb-4">
          <div className="p-4 bg-surface rounded">
            <div className="text-sm text-muted mb-1">Total Columns</div>
            <div className="text-2xl font-bold">{columns.length}</div>
          </div>
          <div className="p-4 bg-surface rounded">
            <div className="text-sm text-muted mb-1">Numeric</div>
            <div className="text-2xl font-bold">{numericCols.length}</div>
          </div>
          <div className="p-4 bg-surface rounded">
            <div className="text-sm text-muted mb-1">Categorical</div>
            <div className="text-2xl font-bold">{categoricalCols.length}</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="mb-4">Visualizations</h3>
        {eda?.plots ? (
          <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(280px,1fr))', gap:16}}>
            {eda.plots.missingness && (
              <div className="p-3 bg-surface rounded">
                <div className="font-semibold mb-2">Missing Data Pattern</div>
                <img src={`${apiUrl}${eda.plots.missingness}`} style={{width:'100%', borderRadius:8}}/>
              </div>
            )}
            {(eda.plots.histograms||[]).map((p:string, idx: number)=> (
              <div key={p} className="p-3 bg-surface rounded">
                <div className="font-semibold mb-2">Distribution {idx + 1}</div>
                <img src={`${apiUrl}${p}`} style={{width:'100%', borderRadius:8}}/>
              </div>
            ))}
            {(eda.plots.categoricals||[]).map((p:string, idx: number)=> (
              <div key={p} className="p-3 bg-surface rounded">
                <div className="font-semibold mb-2">Categorical {idx + 1}</div>
                <img src={`${apiUrl}${p}`} style={{width:'100%', borderRadius:8}}/>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted">No visualizations generated</p>
        )}
      </div>

      <CollapsibleSection title="Column Details" defaultOpen={false}>
        <div className="card">
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique</th></tr>
              </thead>
              <tbody>
                {columns.map((col: any, idx: number) => (
                  <tr key={idx}>
                    <td className="font-semibold">{col.name}</td>
                    <td><span className="badge">{col.type}</span></td>
                    <td>{col.missing_pct ? `${(col.missing_pct * 100).toFixed(1)}%` : '0%'}</td>
                    <td>{col.unique || 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </CollapsibleSection>
    </>
  )
}

function ModelingSection({ modeling, explain, jobId, apiUrl }: { modeling: any, explain: any, jobId: string, apiUrl: string }) {
  const best = modeling?.best || {}
  
  return (
    <>
      <div className="card">
        <h3 className="mb-4">Best Model: {best.name || 'N/A'}</h3>
        <div className="grid-cols-4 mb-4">
          {modeling.task === 'classification' ? (
            <>
              <div className="p-4 bg-surface rounded">
                <div className="text-sm text-muted mb-1">F1 Score</div>
                <div className="text-2xl font-bold">{best.f1?.toFixed(4) || 'N/A'}</div>
              </div>
              <div className="p-4 bg-surface rounded">
                <div className="text-sm text-muted mb-1">Accuracy</div>
                <div className="text-2xl font-bold">{best.accuracy?.toFixed(4) || 'N/A'}</div>
              </div>
              <div className="p-4 bg-surface rounded">
                <div className="text-sm text-muted mb-1">Precision</div>
                <div className="text-2xl font-bold">{best.precision?.toFixed(4) || 'N/A'}</div>
              </div>
              <div className="p-4 bg-surface rounded">
                <div className="text-sm text-muted mb-1">Recall</div>
                <div className="text-2xl font-bold">{best.recall?.toFixed(4) || 'N/A'}</div>
              </div>
            </>
          ) : (
            <>
              <div className="p-4 bg-surface rounded">
                <div className="text-sm text-muted mb-1">R² Score</div>
                <div className="text-2xl font-bold">{best.r2?.toFixed(4) || 'N/A'}</div>
              </div>
              <div className="p-4 bg-surface rounded">
                <div className="text-sm text-muted mb-1">RMSE</div>
                <div className="text-2xl font-bold">{best.rmse?.toFixed(4) || 'N/A'}</div>
              </div>
              <div className="p-4 bg-surface rounded">
                <div className="text-sm text-muted mb-1">MAE</div>
                <div className="text-2xl font-bold">{best.mae?.toFixed(4) || 'N/A'}</div>
              </div>
            </>
          )}
        </div>
        {best.tuned_threshold && (
          <div className="alert alert-info"><strong>Optimized Threshold:</strong> {best.tuned_threshold.toFixed(4)}</div>
        )}
      </div>

      <div className="card">
        <h3 className="mb-4">Model Explainability</h3>
        {explain?.pdp || explain?.roc || explain?.pr ? (
          <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(300px,1fr))', gap:16}}>
            {explain.roc && (
              <div className="p-3 bg-surface rounded">
                <div className="font-semibold mb-2">ROC Curve</div>
                <img src={`${apiUrl}${explain.roc}`} style={{width:'100%', borderRadius:8}}/>
              </div>
            )}
            {explain.pr && (
              <div className="p-3 bg-surface rounded">
                <div className="font-semibold mb-2">Precision-Recall Curve</div>
                <img src={`${apiUrl}${explain.pr}`} style={{width:'100%', borderRadius:8}}/>
              </div>
            )}
            {(explain.pdp||[]).map((p:string, idx: number)=> (
              <div key={p} className="p-3 bg-surface rounded">
                <div className="font-semibold mb-2">Partial Dependence Plot {idx + 1}</div>
                <img src={`${apiUrl}${p}`} style={{width:'100%', borderRadius:8}}/>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted">No explainability plots available</p>
        )}
      </div>
    </>
  )
}
