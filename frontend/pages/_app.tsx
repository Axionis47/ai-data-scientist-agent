import type { AppProps } from 'next/app'
import '../styles/globals.css'
import '../styles/motion.css'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className="app-root">
      <header className="app-header">
        <div className="container">
          <div className="brand" onClick={() => (window.location.href = '/')}>AI Data Scientist</div>
          <nav>
            <a href="/new">New Analysis</a>
            <a href="/history">History</a>
          </nav>
        </div>
      </header>
      <main className="container" style={{minHeight:'calc(100vh - 120px)', display:'flex', flexDirection:'column', justifyContent:'center'}}>
        <Component {...pageProps} />
      </main>
      <footer className="app-footer">
        <div className="container">Crafted for curious minds. Data to decisions—faster.</div>
      </footer>
    </div>
  )
}

