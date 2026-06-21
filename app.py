"""
ResearchTrace — Open Source Portfolio project
Inspired by: IBM Research / PyData Ireland — Open Source Science Dublin Meetup

Concepts demonstrated:
  - Multi-agent pipelines  (NASA AKD / LangGraph style)
  - Guardrails             (IBM Granite Guardian / risk agent)
  - Reproducible tracing   (ADO — provenance, data-reuse, experiment campaigns)

Supports two free LLM providers, switchable from the UI:
  - Ollama  — fully local, no API key, no cost. Install from https://ollama.com
              and pull a model: `ollama pull llama3.2`
  - Groq    — free hosted API, no credit card required. can get a key at
              https://console.groq.com and set GROQ_API_KEY.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional, Literal

import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Config         

# Ollama (local)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_API = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Groq (free hosted API)
GROQ_API = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "paste your GROQ key here")  # get a free key at https://console.groq.com

DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # "ollama" | "groq"

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ResearchTrace", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic schemas (input / output contracts — ADO-style)   

class RunRequest(BaseModel):
    question: str
    provider: Literal["ollama", "groq"] = DEFAULT_PROVIDER


class AgentStep(BaseModel):
    agent: str
    role: str
    output: str
    duration_ms: int
    status: str  # "ok" | "warning" | "blocked"


class ExperimentRun(BaseModel):
    run_id: str
    question: str
    timestamp: float
    steps: list[AgentStep]
    final_answer: Optional[str]
    total_duration_ms: int
    guardrail_status: str  # "passed" | "caution" | "blocked"
    tags: list[str]
    provider: str = "ollama"
    model: str = ""


# Agent helpers 

async def call_ollama(system: str, user: str, max_tokens: int) -> str:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                OLLAMA_CHAT_API,
                json={
                    "model": OLLAMA_MODEL,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not connect to Ollama at {OLLAMA_HOST}. "
                "Make sure Ollama is running ('ollama serve' or open the Ollama app) "
                f"and that you've pulled the model: 'ollama pull {OLLAMA_MODEL}'"
            ),
        )
    if resp.status_code == 404:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{OLLAMA_MODEL}' not found in Ollama. Run: ollama pull {OLLAMA_MODEL}",
        )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


async def call_groq(system: str, user: str, max_tokens: int) -> str:
    api_key = GROQ_API_KEY
    if not api_key or api_key == "paste your GROQ key here":
        raise HTTPException(
            status_code=500,
            detail=(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                "(no credit card required), then either set the environment variable "
                "GROQ_API_KEY, or paste it into the GROQ_API_KEY constant near the top "
                "of app.py."
            ),
        )
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                GROQ_API,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": GROQ_MODEL,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Could not reach Groq API. Check your internet connection.")
    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Groq API key was rejected. Check GROQ_API_KEY is correct.")
    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="Groq free-tier rate limit hit. Wait a minute and try again.")
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_llm(system: str, user: str, max_tokens: int = 600, provider: str = "ollama") -> str:
    """Single LLM call — the atom of every agent. Routes to Ollama (local) or Groq (free hosted API)."""
    if provider == "groq":
        return await call_groq(system, user, max_tokens)
    return await call_ollama(system, user, max_tokens)


async def run_planner(question: str, provider: str) -> tuple[str, int]:
    t0 = time.monotonic()
    output = await call_llm(
        "You are a research planner. Decompose the user's question into exactly 3 "
        "numbered sub-questions a researcher should answer. Respond with ONLY the "
        "numbered list, no preamble.",
        question,
        max_tokens=300,
        provider=provider,
    )
    return output, int((time.monotonic() - t0) * 1000)


async def run_researcher(question: str, plan: str, provider: str) -> tuple[str, int]:
    t0 = time.monotonic()
    output = await call_llm(
        "You are a research agent. Given a question and a research plan, answer each "
        "sub-question concisely (2-3 sentences each). Format: Q1: ...\nA1: ...",
        f"Main question: {question}\n\nPlan:\n{plan}",
        max_tokens=700,
        provider=provider,
    )
    return output, int((time.monotonic() - t0) * 1000)


async def run_guardrail(question: str, research: str, provider: str) -> tuple[str, int, str]:
    """
    Guardrail agent — mirrors IBM Granite Guardian / risk agent concept.
    Returns (output, duration_ms, status)
    """
    t0 = time.monotonic()
    output = await call_llm(
        "You are a guardrail agent. Review the research and output EXACTLY this format:\n"
        "CONFIDENCE: High | Medium | Low\n"
        "FLAGS: <one sentence, or 'None'>\n"
        "RISK: <one sentence, or 'None'>\n"
        "RECOMMENDATION: Proceed | Proceed with caution | Do not proceed",
        f"Question: {question}\n\nResearch:\n{research}",
        max_tokens=200,
        provider=provider,
    )
    duration = int((time.monotonic() - t0) * 1000)
    lower = output.lower()
    if "do not proceed" in lower:
        status = "blocked"
    elif "caution" in lower:
        status = "caution"
    else:
        status = "passed"
    return output, duration, status


async def run_synthesiser(question: str, research: str, guardrail: str, provider: str) -> tuple[str, int]:
    t0 = time.monotonic()
    output = await call_llm(
        "You are a synthesiser. Write a clear, well-structured final answer (3-4 paragraphs) "
        "based on the research. Do not mention the pipeline or agents.",
        f"Question: {question}\n\nResearch:\n{research}\n\nGuardrail notes:\n{guardrail}",
        max_tokens=800,
        provider=provider,
    )
    return output, int((time.monotonic() - t0) * 1000)


# Provenance / persistence (ADO-style data tracking)

def save_run(run: ExperimentRun) -> None:
    path = EXPERIMENTS_DIR / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2))


def load_all_runs() -> list[ExperimentRun]:
    runs = []
    for f in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
        try:
            runs.append(ExperimentRun.model_validate_json(f.read_text()))
        except Exception:
            pass
    return runs


def load_run(run_id: str) -> ExperimentRun:
    path = EXPERIMENTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return ExperimentRun.model_validate_json(path.read_text())


# API routes

@app.post("/api/run", response_model=ExperimentRun)
async def run_pipeline(req: RunRequest):
    """
    Execute the full multi-agent pipeline and persist the run.
    Pipeline: Planner → Researcher → Guardrail → Synthesiser
    Provider: "ollama" (local, free) or "groq" (hosted, free tier)
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    provider = req.provider
    model_name = OLLAMA_MODEL if provider == "ollama" else GROQ_MODEL

    run_id = str(uuid.uuid4())[:8]
    steps: list[AgentStep] = []
    wall_start = time.monotonic()

    # Step 1 — Planner
    plan, ms = await run_planner(req.question, provider)
    steps.append(AgentStep(agent="planner", role="Decomposes question into sub-questions",
                            output=plan, duration_ms=ms, status="ok"))

    # Step 2 — Researcher
    research, ms = await run_researcher(req.question, plan, provider)
    steps.append(AgentStep(agent="researcher", role="Answers each sub-question with facts",
                            output=research, duration_ms=ms, status="ok"))

    # Step 3 — Guardrail
    guardrail_out, ms, g_status = await run_guardrail(req.question, research, provider)
    steps.append(AgentStep(agent="guardrail", role="Checks for risk, bias, and confidence",
                            output=guardrail_out, duration_ms=ms, status=g_status))

    # Step 4 — Synthesiser (blocked if guardrail says so)
    final_answer: Optional[str] = None
    if g_status != "blocked":
        synth_out, ms = await run_synthesiser(req.question, research, guardrail_out, provider)
        final_answer = synth_out
        steps.append(AgentStep(agent="synthesiser", role="Produces the final answer",
                                output=synth_out, duration_ms=ms, status="ok"))
    else:
        steps.append(AgentStep(agent="synthesiser", role="Produces the final answer",
                                output="Blocked by guardrail agent.", duration_ms=0,
                                status="blocked"))

    total_ms = int((time.monotonic() - wall_start) * 1000)

    # Auto-tag based on content (lightweight provenance metadata)
    tags = []
    q_lower = req.question.lower()
    for keyword, tag in [("agent", "agents"), ("llm", "llm"), ("science", "science"),
                         ("nasa", "space"), ("climate", "climate"), ("model", "ml"),
                         ("data", "data"), ("research", "research")]:
        if keyword in q_lower:
            tags.append(tag)

    run = ExperimentRun(
        run_id=run_id,
        question=req.question,
        timestamp=time.time(),
        steps=steps,
        final_answer=final_answer,
        total_duration_ms=total_ms,
        guardrail_status=g_status,
        tags=list(set(tags)) or ["general"],
        provider=provider,
        model=model_name,
    )
    save_run(run)
    return run


@app.get("/api/runs", response_model=list[ExperimentRun])
async def list_runs():
    """Return all past experiment runs — provenance history."""
    return load_all_runs()


@app.get("/api/runs/{run_id}", response_model=ExperimentRun)
async def get_run(run_id: str):
    """Return a single run by ID."""
    return load_run(run_id)


@app.get("/api/runs/{run_id}/compare/{other_id}")
async def compare_runs(run_id: str, other_id: str):
    """Compare two runs on the same question — ADO-style data reuse check."""
    a = load_run(run_id)
    b = load_run(other_id)
    return {
        "run_a": {"id": a.run_id, "question": a.question,
                  "guardrail": a.guardrail_status, "ms": a.total_duration_ms},
        "run_b": {"id": b.run_id, "question": b.question,
                  "guardrail": b.guardrail_status, "ms": b.total_duration_ms},
        "same_question": a.question.strip().lower() == b.question.strip().lower(),
    }


@app.get("/api/health")
async def health_check():
    """Check Ollama (local) and Groq (hosted) availability — drives the UI provider toggle."""
    result = {
        "ollama": {"running": False, "model": OLLAMA_MODEL, "model_ready": False, "available_models": []},
        "groq": {"key_set": bool(GROQ_API_KEY and GROQ_API_KEY != ""), "model": GROQ_MODEL},
        "default_provider": DEFAULT_PROVIDER,
    }
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        result["ollama"]["running"] = True
        result["ollama"]["available_models"] = models
        result["ollama"]["model_ready"] = any(OLLAMA_MODEL in m for m in models)
    except Exception:
        pass
    return result


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(
        content=Path("static/index.html").read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
    )