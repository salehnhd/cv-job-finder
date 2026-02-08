# CV Job Finder (CrewAI)

Paste your CV (DOCX) → run one command → get a ranked list of job links + fit scores + gaps.

**Outputs**
- `shortlisted_jobs.md` (easy to read)
- `shortlisted_jobs.json` (structured)
- `crew_output.txt` (full logs)

---

## Quick Start (Windows, 5 minutes)

### 1) Download this project
- Click **Code → Download ZIP**
- Extract it
- Open PowerShell in the extracted folder

### 2) Create and activate a virtual environment
```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Set your API keys
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:SERPER_API_KEY="..."
```

### 4) Add your CV
Put your CV in the project folder and name it exactly: "cvLLM.docx"

### 5) Run
python cv_job_finder.py


### What it does:

1- Reads your CV from .docx.

2- An AI agent extracts your profile (skills, domains, seniority).

3- Another AI agent generates job search queries.

4- Searches job postings via Serper.

5- Scrapes job pages.

6- The last AI agent scores each job vs your CV and ranks the best matches.


### Privacy & Safety

1- Your CV file stays on your machine.

2- The script sends text to:

- OpenAI (LLM reasoning)

- Serper (job search)

- Do not upload your CV to GitHub.

- .gitignore prevents committing .docx, .env, and outputs.


### Troubleshooting

- Python not found / wrong version

```powershell
py -0p
```

- Make sure Python 3.12 is installed.

- Missing keys: Ensure OPENAI_API_KEY and SERPER_API_KEY are set (or .env exists).



### Disclaimer 
This is an experimental tool. Always verify job postings and requirements manually.


