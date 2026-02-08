import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


# ----------------------------
# Config
# ----------------------------
CV_FILENAME = "cvLLM.docx"
MAX_JOBS_TO_EVALUATE = 12          # keep small to control time/cost
REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CV-Job-Finder/1.0"

# Default model (you can change in env: OPENAI_MODEL)
DEFAULT_MODEL = "gpt-4o-mini"


# ----------------------------
# Helpers
# ----------------------------
def die(msg: str) -> None:
    raise SystemExit(f"\n❌ {msg}\n")


def read_docx_as_text(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120]


def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None


def extract_main_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "noscript", "svg", "img", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Prefer <main> if present
    main = soup.find("main")
    text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
    text = clean_text(text)

    # Keep reasonable amount
    return text[:9000]


def serper_search(api_key: str, query: str, num: int = 10) -> List[Dict[str, Any]]:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num}

    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    out = []
    for item in data.get("organic", [])[:num]:
        out.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return out


# ----------------------------
# “Agent” schemas (Pydantic)
# ----------------------------
class CVProfile(BaseModel):
    target_roles: List[str] = Field(default_factory=list)
    seniority: str = Field(default="unknown")
    core_skills: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class SearchPlan(BaseModel):
    queries: List[str] = Field(default_factory=list)


class JobFit(BaseModel):
    score: float = Field(ge=0, le=10)
    reasoning: str
    gaps: List[str] = Field(default_factory=list)
    recommended_keywords: List[str] = Field(default_factory=list)


# ----------------------------
# OpenAI client + “agents”
# ----------------------------
def get_openai_client() -> Tuple[OpenAI, str]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        die("Missing OPENAI_API_KEY. Set it in your shell or in a .env file.")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return OpenAI(api_key=api_key), model


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def llm_json(client: OpenAI, model: str, system: str, user: str) -> Dict[str, Any]:
    # Uses JSON mode-like prompting; robust enough for this use.
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    # Attempt to find JSON object in response
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def agent_cv_analyst(client: OpenAI, model: str, cv_text: str) -> CVProfile:
    system = (
        "You are a CV Analyst. Extract a concise structured profile from a CV. "
        "Return ONLY valid JSON with keys: target_roles (list of strings), seniority (string), "
        "core_skills (list), domains (list), keywords (list). No extra keys."
    )
    user = (
        "CV TEXT:\n"
        f"{cv_text}\n\n"
        "Return JSON only."
    )
    data = llm_json(client, model, system, user)
    return CVProfile(**data)


def agent_search_planner(client: OpenAI, model: str, profile: CVProfile) -> SearchPlan:
    system = (
        "You are a Job Search Planner. Create effective Google-style queries to find real job postings. "
        "Return ONLY valid JSON with key: queries (list of strings). Make 6-10 queries."
    )
    user = (
        "PROFILE JSON:\n"
        f"{profile.model_dump_json(indent=2)}\n\n"
        "Constraints:\n"
        "- Prefer queries that return job postings (not advice articles)\n"
        "- Include role + domain keywords\n"
        "- Use site: filters only if helpful (optional)\n\n"
        "Return JSON only."
    )
    data = llm_json(client, model, system, user)
    return SearchPlan(**data)


def agent_fit_evaluator(client: OpenAI, model: str, profile: CVProfile, job_title: str, job_text: str, job_url: str) -> JobFit:
    system = (
        "You are a Job Fit Evaluator. Score how well the job matches the candidate profile. "
        "Return ONLY valid JSON with keys: score (0-10), reasoning (string), gaps (list), recommended_keywords (list). "
        "Be specific and honest. No extra keys."
    )
    user = (
        "CANDIDATE PROFILE:\n"
        f"{profile.model_dump_json(indent=2)}\n\n"
        f"JOB TITLE: {job_title}\n"
        f"JOB URL: {job_url}\n\n"
        "JOB DESCRIPTION (clean text):\n"
        f"{job_text}\n\n"
        "Return JSON only."
    )
    data = llm_json(client, model, system, user)
    return JobFit(**data)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    load_dotenv()

    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    if not serper_key:
        die("Missing SERPER_API_KEY. Set it in your shell or in a .env file.")

    # Read CV
    if not os.path.exists(CV_FILENAME):
        die(f"Missing {CV_FILENAME}. Put your CV in this folder and name it exactly: {CV_FILENAME}")

    cv_text = read_docx_as_text(CV_FILENAME)
    if not cv_text:
        die("Could not extract text from the DOCX. Is it empty or image-only?")

    # Save extracted text
    with open("cv_text.txt", "w", encoding="utf-8") as f:
        f.write(cv_text)

    # OpenAI
    client, model = get_openai_client()

    logs: List[str] = []
    logs.append(f"Model: {model}")
    logs.append("Step 1/4: Building profile from CV...")

    profile = agent_cv_analyst(client, model, cv_text)
    logs.append(f"Profile: {profile.model_dump()}")

    logs.append("Step 2/4: Creating search queries...")
    plan = agent_search_planner(client, model, profile)
    queries = [q for q in plan.queries if q.strip()]
    if not queries:
        die("Search planner produced no queries.")

    logs.append("Queries:")
    logs.extend([f"- {q}" for q in queries])

    # Search + collect URLs
    logs.append("Step 3/4: Searching job postings (Serper)...")
    seen = set()
    candidates: List[Dict[str, Any]] = []

    for q in queries[:10]:
        try:
            results = serper_search(serper_key, q, num=8)
        except Exception as e:
            logs.append(f"[WARN] Serper failed for query: {q} -> {e}")
            continue

        for item in results:
            url = (item.get("link") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            candidates.append(item)

        # small delay to be polite
        time.sleep(0.3)

        if len(candidates) >= MAX_JOBS_TO_EVALUATE:
            break

    if not candidates:
        die("No job URLs found. Try different queries or check your SERPER_API_KEY.")

    # Scrape and evaluate
    logs.append("Step 4/4: Scraping and scoring jobs...")
    scored: List[Dict[str, Any]] = []

    for item in candidates[:MAX_JOBS_TO_EVALUATE]:
        title = item.get("title") or "Untitled job"
        url = item.get("link") or ""
        snippet = item.get("snippet") or ""

        html = fetch_html(url)
        if not html:
            logs.append(f"[SKIP] Could not fetch: {url}")
            continue

        job_text = extract_main_text_from_html(html)
        if len(job_text) < 200:
            logs.append(f"[SKIP] Too little text extracted: {url}")
            continue

        try:
            fit = agent_fit_evaluator(client, model, profile, title, job_text, url)
        except Exception as e:
            logs.append(f"[WARN] Fit evaluator failed for {url}: {e}")
            continue

        scored.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "score": float(fit.score),
                "reasoning": fit.reasoning,
                "gaps": fit.gaps,
                "recommended_keywords": fit.recommended_keywords,
            }
        )
        logs.append(f"[OK] {fit.score:.1f}/10 — {title}")

    if not scored:
        die("No jobs could be scored (scraping/LLM issues). Try again or reduce filters.")

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Output JSON
    with open("shortlisted_jobs.json", "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    # Output Markdown
    lines = []
    lines.append("# Shortlisted Jobs (Ranked)\n")
    lines.append("| Rank | Score | Job Title | Link | Notes |")
    lines.append("|------|------:|----------|------|-------|")

    for i, job in enumerate(scored, start=1):
        title = job["title"].replace("|", "\\|")
        url = job["url"]
        score = f'{job["score"]:.1f}'
        notes = clean_text(job["reasoning"])[:140].replace("|", "\\|")
        lines.append(f"| {i} | {score} | {title} | [link]({url}) | {notes} |")

    lines.append("\n## Gaps to Address (from top matches)\n")
    all_gaps = []
    for job in scored[:5]:
        for g in job.get("gaps", []):
            if g and g not in all_gaps:
                all_gaps.append(g)

    if all_gaps:
        for g in all_gaps[:20]:
            lines.append(f"- {g}")
    else:
        lines.append("- (No major gaps detected.)")

    with open("shortlisted_jobs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Logs
    with open("crew_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    print("\n✅ Done!")
    print("Created:")
    print("- shortlisted_jobs.md")
    print("- shortlisted_jobs.json")
    print("- crew_output.txt")
    print("- cv_text.txt")


if __name__ == "__main__":
    main()
