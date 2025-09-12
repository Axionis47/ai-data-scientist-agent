# Frontend files documentation

- next.config.js
  - Purpose: Next.js config; rewrites to backend API if applicable

- pages/index.tsx or app/page.tsx (depending on structure)
  - Purpose: Landing/results UI
  - Interacts with: /upload, /analyze, /status, /result endpoints

- components/Upload.tsx
  - Purpose: File selection / drag-drop; POST to /upload

- components/AnalyzeButton.tsx
  - Purpose: Trigger /analyze with selected job

- components/Results.tsx
  - Purpose: Poll /status; fetch /result and render report_html

- lib/api.ts
  - Purpose: Small wrapper for fetch calls to backend API; uses NEXT_PUBLIC_API_URL

Note: File names may vary depending on the current Next.js app folder structure; adjust accordingly.

