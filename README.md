# CV Job Finder (AI Agent–based)

A lightweight Python tool that uses **LLM-based AI agents** to analyze a CV, search real job postings, and rank jobs by relevance.

The agents reason over your CV and job descriptions using **prompt-based analysis (RAG-style, without vector databases)**.

---

## What It Does

- Reads a CV (`.docx`) locally
- AI agent extracts skills, domains, and seniority
- AI agent generates job search queries
- Finds real job postings via Google (Serper API)
- Scrapes job descriptions
- AI agent evaluates job–CV fit with reasoning and gaps
- Outputs ranked job matches

---

## AI Architecture (Simplified)

- **CV Analyst Agent** – builds a structured profile from the CV  
- **Search Planner Agent** – designs job search queries  
- **Fit Evaluator Agent** – scores relevance and highlights gaps  

Agents are implemented as **plain Python functions calling an LLM**  
(no CrewAI framework, no vector database).

---

## Requirements

- Python **3.12**
- OpenAI API key
- Serper API key

---

## Installation

```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## API Keys
```bash
$env:OPENAI_API_KEY="sk-..."
$env:SERPER_API_KEY="your-serper-key"
```

---

## Prepare CV

Place your CV in the project folder as **`cvLLM.docx`**

---

## Run
```bash
python cv_job_finder.py
```

---

## Output

- `shortlisted_jobs.md` – ranked jobs

- `shortlisted_jobs.json` – structured results

- `crew_output.txt` – agent reasoning/logs

- `cv_text.txt` – extracted CV text

- All outputs are generated locally.

---

## Privacy

- CV processed locally

- No data stored or retained

- External calls: OpenAI (LLM), Serper (job search)

---

## License

MIT

