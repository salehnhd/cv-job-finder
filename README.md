# CV Job Finder (CrewAI)

A Python tool built with CrewAI that analyzes a CV in `.docx` format, searches for relevant job postings online, scrapes job descriptions, and ranks how well each job matches the CV using LLM-based agents.

---

## Requirements

You need:

- **Python 3.12**
- An **OpenAI API key**
- A **Serper API key** (used for job search)

---

## Installation (Windows)

Open **PowerShell**, navigate to the project folder, and run:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# Configure API Keys

In the same PowerShell window, set your API keys:
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:SERPER_API_KEY="..."
```

# Prepare Your CV

Place your CV file in the project folder.

Name the file exactly:
```powershell
cvLLM.docx
```

# Run the Script

With the virtual environment still active, run:
```powershell
python cv_job_finder.py
```

# Output Files

After the script finishes, the following files will be created:

-shortlisted_jobs.md – human-readable table of best-matching jobs

-shortlisted_jobs.json – structured job data

-crew_output.txt – full agent reasoning and logs




# Notes

-CrewAI memory and RAG are disabled to avoid heavy dependencies on Windows.

-Your CV is processed locally and is not stored or uploaded.

-Only OpenAI and Serper APIs are called externally.




# Troubleshooting

If the script fails:

-Make sure Python 3.12 is installed and active

-Ensure the virtual environment is activated

-Verify both API keys are set

-Check that the CV filename is exactly cvLLM.docx



# Disclaimer

-This project is provided for experimentation and personal use.
-Always verify job postings manually before applying.
