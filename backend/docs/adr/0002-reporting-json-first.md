# ADR 0002: Reporting JSON-first with Fallback

## Context
LLM-based HTML generation can vary; we want a reliable, structured path that can be validated and transformed into deterministic HTML.

## Decision
- Add REPORT_JSON_FIRST feature flag that asks the LLM to return a JSON structure (title, kpis, sections)
- Validate against a lightweight schema; render deterministic HTML if valid
- On failure, fall back to the existing HTML prompt, then to a deterministic local template

## Consequences
- Better reliability and easier testing of report content
- Debuggability via explicit validation errors
- Slight added complexity in reporting code; governed by a feature flag

