import React, { useRef, useState } from 'react'

export function FileDropZone({ onFile }: { onFile: (f: File) => void }){
  const inputRef = useRef<HTMLInputElement>(null)
  const [hover, setHover] = useState(false)
  return (
    <div className="dropzone" style={{minHeight:140}} onClick={()=>inputRef.current?.click()} onDragOver={e=>{e.preventDefault(); setHover(true)}} onDragLeave={()=>setHover(false)} onDrop={e=>{e.preventDefault(); setHover(false); if(e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0])}} aria-label="Upload data">
      <div style={{opacity:.9}}>
        <div style={{fontWeight:700, marginBottom:8, fontSize:16}}>Drop your data here</div>
        <div style={{color:'#9ca3af'}}>CSV, JSON, Parquet. Or click to select.</div>
      </div>
      <input ref={inputRef} type="file" style={{display:'none'}} onChange={e=>{const f=e.target.files?.[0]; if(f) onFile(f)}} />
    </div>
  )
}

