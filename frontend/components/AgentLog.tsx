import React from 'react'

type Msg = { role: string, content: string }
export function AgentLog({ messages }: { messages: Msg[] }){
  return (
    <div className="log" aria-live="polite">
      {messages?.length ? messages.map((m,i)=> (
        <div key={i} style={{padding:'4px 0'}}>
          <span className="badge" style={{marginRight:8}}>{m.role}</span>
          <span>{m.content}</span>
        </div>
      )) : <div className="skeleton" style={{height:80,borderRadius:8}}/>}
    </div>
  )
}

