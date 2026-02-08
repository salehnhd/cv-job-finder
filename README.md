# CV Job Finder

A lightweight Python tool that uses a **large language model (LLM)** to analyze a CV, search real job postings, and rank jobs by relevance.

The tool combines LLM-based reasoning with web search and basic scraping to compare a candidate profile with job descriptions.

---

## What It Does

- Reads a CV (`.docx`) locally  
- Extracts skills, domains, and seniority using an LLM  
- Generates job search queries  
- Finds real job postings via Google (Serper API)  
- Scrapes job descriptions from the web  
- Evaluates job–CV fit with scores, reasoning, and gaps  
- Outputs ranked job matches  

---

## How It Works

The workflow is organized into three logical steps:

- **CV Analysis** – builds a structured profile from the CV  
- **Search Planning** – creates job search queries from the profile  
- **Job Evaluation** – scores each job against the profile  

Each step is implemented as a Python function calling the same LLM model with a different prompt.

---

## Techniques Used

- Prompt-based structured extraction (JSON outputs)  
- Web search via Serper API  
- HTML scraping with BeautifulSoup  
- Deterministic Python orchestration  

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
Place your CV in the project folder as (`cvLLM.docx`)

---

## Run
`python cv_job_finder.py`

---

## Output
The script generates the following files:

`shortlisted_jobs.md` – ranked jobs (Markdown)

`shortlisted_jobs.json` – structured results

`crew_output.txt` – execution logs

`cv_text.txt` – extracted CV text

---

## Privacy

- CV is processed locally

- External calls are limited to:

- OpenAI (LLM reasoning)

- Serper (job search)

## License
MIT


