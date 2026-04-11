# Agent Swarm

A general-purpose agent swarm built with **LangGraph Functional API**, **browser-use**, and **LangSmith** for trajectory visualization. The manager LLM runs a tool-call loop and can browse the web live using a real Chrome browser driven by Gemini.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Manager Agent  (tool-call loop)                    │
│                                                     │
│  Tools: browse_web, (spawn_sub_agent*)              │
│                                                     │
│  1. Receives user query + system prompt             │
│  2. Decides which tools to call and how many times  │
│  3. Compiles results into a final answer            │
└───────────────────────┬─────────────────────────────┘
                        │  browse_web × N (parallel)
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      browser-use   browser-use  browser-use
      session 1     session 2    session N
      (Chrome)      (Chrome)     (Chrome)
           │            │            │
           └────────────┴────────────┘
                        │  results fan-in
                        ▼
                  Final Answer
```

> *`spawn_sub_agent` is defined but currently not bound to the manager. Re-add it to `tools` in `agent_swarm` when you need parallel sub-agent delegation.

### Key components

| Component | Description |
|---|---|
| `agent_swarm` | `@entrypoint` — the manager's tool-call agent loop |
| `browse_web` | LangChain `@tool` — opens a real Chrome window, drives it with Gemini via browser-use, and records the session |
| `spawn_sub_agent` | LangChain `@tool` — spawns a sub-agent with its own tool loop (has access to `browse_web`) |
| `run_tool_call` | LangGraph `@task` — wraps each tool invocation for parallel execution and LangSmith tracing |
| `_on_browser_step` | Callback that prints live step info (URL, goal, action) to the terminal at each browser-use step |

### How `browse_web` works

browser-use drives a **real Chrome window** (your system Chrome, not Playwright's sandboxed Chromium). Gemini receives raw page state — DOM structure and a screenshot — and decides every browser action (click, scroll, type, navigate). No browser-use cloud API key is used.

```
browse_web(task)
    │
    ▼
BrowserAgent (browser-use)
    ├── LLM: ChatGoogle (Gemini via google.genai)
    ├── Browser: system Chrome, headless=False
    ├── Profile: your real Chrome profile (cookies/sessions copied to temp dir)
    ├── demo_mode=True  →  agent panel overlaid in browser window
    └── register_new_step_callback  →  live terminal output per step

Each run saves to recordings/<YYYYMMDD_HHMMSS>/
    ├── *.mp4        full-fidelity screen recording (CDP screencast)
    ├── session.gif  animated GIF of LLM-visible screenshots
    └── actions.json complete action history (goals, actions, results)
```

### How parallelism works

When the manager LLM returns multiple `tool_calls` in one response, all are dispatched as `@task` instances before any `.result()` is awaited — so they run concurrently.

```
manager LLM ──► [browse_web_0, browse_web_1, ..., browse_web_N]
                      │              │                   │
                   @task           @task               @task    ← all dispatched first
                      │              │                   │
                      └──────────────┴───────────────────┘
                                     │ fan-in
                               final answer
```

## Setup

**1. Install dependencies**

```bash
cd agent-swarm
uv sync
```

**2. Install Playwright browsers** (first time only)

```bash
uv run python -m playwright install chromium
```

**3. Configure environment**

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `LANGSMITH_API_KEY` | API key from your [LangSmith](https://smith.langchain.com) account |
| `LANGSMITH_PROJECT` | LangSmith project name (default: `agent-swarm`) |
| `LANGSMITH_TRACING` | Set to `true` to enable tracing |

## Running

```bash
uv run python main.py
```

A Chrome window will open. You can watch the agent scroll and click in real time. The demo panel in the browser shows the agent's current goal and last action. Each step is also printed to the terminal.

Recordings are saved automatically to `recordings/<timestamp>/`.

## Extending

- **Change the task**: edit `BROWSER_USE_QUERY` in `prompts.py`
- **Change the system prompt**: edit `BROWSER_USE_SYSTEM_PROMPT` in `prompts.py`
- **Re-enable sub-agent spawning**: add `spawn_sub_agent` back to the `tools` list in `agent_swarm`
- **Use a different Chrome profile**: change `profile_directory` in the `BrowserProfile` (e.g. `"Profile 1"`)
- **Add dependencies**: `uv add <package>`
