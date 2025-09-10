import { useEffect, useState } from 'react'

// Placeholder history; for now, we can store recent jobIds in localStorage
export default function History(){
  const [jobs, setJobs] = useState<string[]>([])
  useEffect(()=>{
    const j = JSON.parse(localStorage.getItem('jobs') || '[]'); setJobs(j)
  },[])
  return (
    <div style={{padding:'24px 0'}}>
      <h1>History</h1>
      {jobs.length ? (
        <ul>
          {jobs.map(id => (<li key={id}><a href={`/results/${id}`}>{id}</a></li>))}
        </ul>
      ) : <p>No runs yet.</p>}
    </div>
  )
}

