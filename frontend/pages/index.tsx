export default function Home(){
  return (
    <div style={{padding:'80px 0'}}>
      <div className="card" style={{textAlign:'center', padding:'32px'}}>
        <h1 style={{fontSize:40, marginBottom:12}}>Your AI Data Scientist</h1>
        <p style={{color:'#9ca3af',margin:'0 auto 20px', maxWidth:680}}>From messy datasets to crisp decisions. Surface patterns, test hypotheses, and model outcomes—no code required.</p>
        <div className="row" style={{justifyContent:'center'}}>
          <a href="/new" className="btn">Start New Analysis</a>
          <a href="/history" className="btn secondary">View History</a>
        </div>
      </div>
      <div style={{height:16}} />
      <div className="card" style={{padding:'20px'}}>
        <div style={{marginBottom:8, fontWeight:700}}>Try a sample</div>
        <div style={{color:'#9ca3af'}}>Start with a tiny Titanic dataset to see profiling, modeling, and QA in action. Path: <code>backend/data/sample/titanic_small.csv</code></div>
      </div>
    </div>
  )
}

