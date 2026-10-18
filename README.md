# Kemira GEO Audit Agent (prototype)

An agent that checks how an AI assistant describes a company and how the
company's website supports AI readability. Built with Python and the OpenAI API.

## What it does
- Asks an assistant questions about Kemira and compares answers with a reference fact sheet
- Inspects kemira.com pages (headings, JSON-LD types, FAQ signals, links)
- Drafts a report, then a second pass reviews it against the collected evidence

## Guardrails
- Only kemira.com pages; only URLs seen in earlier link lists
- Public information only, no personal data
- Reports are AI-generated drafts and need human review

## Limitations
- The assistant is a plain model, not a search-enabled one
- Small sample; only JSON-LD is checked; answers vary between runs

## Setup
pip install -r requirements.txt
Copy .env.example to .env and add your OpenAI API key
python geo_agent_v2.py

Independent prototype, not affiliated with Kemira.