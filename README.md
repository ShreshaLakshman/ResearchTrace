# ResearchTrace

An **open-source** portfolio project inspired by the **Open Source Science Dublin Meetup** (IBM Research × PyData Ireland), built after attending the event to test whether I'd actually understood the core ideas from the talks.

# Motive of the project
What does it take to build a guardrailed, reproducible multi-agent research pipeline using only free and open tools?

## What this project is

The meetup covered three talks: NASA's multi-agent research platform, IBM's approach to taking AI agents from prototype to production (with guardrails), and IBM's `ado` framework for reproducible agent experimentation. Rather than just taking notes, I built a small working system that touches all three ideas at once:

A question goes through a **4-agent pipeline** - a planner breaks it into sub-questions, a researcher answers them, a guardrail agent checks the research for risk or low confidence (and can block the pipeline outright), and a synthesiser writes the final answer. Every step is timed, logged, and saved to disk so past runs can be reviewed or compared later -that's the reproducibility piece from the `ado` talk.

Supports **two free LLM providers**, switchable right in the UI:

| Provider | Cost | Setup | Speed |
|---|---|---|---|
| **Ollama** | Free, fully local | Install app + pull a model | Depends on your machine |
| **Groq** | Free tier, hosted | Free API key, no credit card | Very fast (LPU hardware) |

No paid API key is ever required.

Demonstrates three concepts from the talks in one working app:

| Concept | Talk | Where in this project |
|---|---|---|
| Multi-agent pipeline | NASA AKD (James Barry) | Planner → Researcher → Guardrail → Synthesiser chain |
| Guardrails | IBM Granite Guardian (Fabio Lorenzi) | Guardrail agent blocks or flags the synthesiser |
| Reproducible tracing | ADO framework (Michael Johnston) | Every run saved as JSON with full provenance |

---

## Quickstart

### 1. Unzip and enter the project folder

```bash
cd researchtrace
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up at least one provider

You only need **one** of these - pick whichever suits you. You can also set up both and switch between them live in the UI.

#### Option A - Ollama (fully local, no key)

1. Install from **https://ollama.com**
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Make sure Ollama is running (it usually auto-starts; on Linux run `ollama serve`)

#### Option B - Groq (free hosted API)

1. Sign up free at **https://console.groq.com** (no credit card)
2. Create an API key
3. Set it as an environment variable:
   ```bash
   set GROQ_API_KEY=gsk_... 
   ```

### 5. Run the app

```bash
uvicorn app:app --reload
```

### 6. Open it

Go to **http://localhost:8000** - use the toggle next to "run pipeline" to switch between **ollama (local)** and **groq (free api)**.

---

## Choosing models

```bash
# Ollama - use any model you've pulled
export OLLAMA_MODEL=mistral
ollama pull mistral

# Groq - pick from their free model list
export GROQ_MODEL=llama-3.1-8b-instant   # faster, lighter
# or
export GROQ_MODEL=llama-3.3-70b-versatile  # default, stronger
```

```bash
# Set the default provider used on first load (the UI toggle can still override it)
export LLM_PROVIDER=groq   # or "ollama" (default)
```

---

## Project structure

```
researchtrace/
├── app.py              # FastAPI backend - both providers + agents + API routes
├── requirements.txt
├── static/
│   └── index.html      # Single-page frontend with the provider toggle
└── experiments/        # Auto-created - one JSON file per run, tagged with provider used
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/run` | Run the full pipeline. Body: `{"question": "...", "provider": "ollama" \| "groq"}` |
| `GET` | `/api/runs` | List all past runs |
| `GET` | `/api/runs/{id}` | Get a single run |
| `GET` | `/api/runs/{id}/compare/{other_id}` | Compare two runs (ADO-style) |
| `GET` | `/api/health` | Check Ollama + Groq readiness - drives the UI toggle dots |

### Example with curl

```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangGraph and why is it useful?", "provider": "groq"}'
```

---

## How the pipeline works
<img width="457" height="379" alt="image" src="https://github.com/user-attachments/assets/a351564d-4b59-4f9b-8455-e351171fdc02" />




All four agents call the **same `call_llm()` function** - they differ only in system prompt and role, and the provider can be swapped per-run without touching the agent logic. This mirrors the orchestration pattern from the NASA AKD talk: the architecture is what matters, not which model sits underneath it.

---

## Concepts mapped to code

- **`call_llm()` / `call_ollama()` / `call_groq()`** - provider-agnostic dispatch; every agent is just a different system prompt over whichever backend is selected
- **`run_planner()`** - agentic decomposition (NASA AKD style)
- **`run_guardrail()`** - risk agent with BLOCK capability (IBM Granite Guardian)
- **`save_run()` / `load_all_runs()`** - provenance + data reuse (ADO), now also recording which provider/model produced each run
- **`/api/runs/{id}/compare/{other_id}`** - run comparison (ADO experiment campaigns)
- **`/api/health`** - reports readiness of both providers so the UI toggle can show live status dots
- **Pydantic `ExperimentRun` schema** - structured input/output contracts

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not connect to Ollama" | Run `ollama serve`, or open the Ollama desktop app |
| "Model not found" (Ollama) | Run `ollama pull llama3.2` (or whichever model you set) |
| "GROQ_API_KEY environment variable not set" | Get a free key at console.groq.com and `export GROQ_API_KEY=...` |
| "Groq API key was rejected" | Double-check you copied the full key, including the `gsk_` prefix |
| "Groq free-tier rate limit hit" | Wait a minute - free tier allows ~30 requests/min - then retry |
| Ollama responses are slow | Normal on CPU-only machines - try a smaller model like `phi3` or `llama3.2:1b`, or switch to Groq in the UI |



*Developed as a learning project to master AI-augmented workflows. All core concepts, system architecture, and debugging were performed manually, while AI tools were utilized strictly for generating basic code structure and accelerating implementation*
