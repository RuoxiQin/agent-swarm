# Agent Swarm

A general-purpose parallel agent swarm built with **LangGraph Functional API** and **LangSmith** for trajectory visualization. The manager LLM autonomously decomposes a user request into sub-tasks and dispatches them to sub-agents in parallel.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Manager Agent  (tool-call loop)                    │
│                                                     │
│  1. Receives user query + MANAGER_SYSTEM_PROMPT     │
│  2. Decides how many sub-tasks to spawn             │
│  3. Calls `spawn_sub_agent` tool N times            │
│  4. Compiles results into final answer              │
└───────────────────────┬─────────────────────────────┘
                        │  spawn_sub_agent × N
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      Sub-agent 1  Sub-agent 2  Sub-agent N
      (parallel)   (parallel)   (parallel)
           │            │            │
           └────────────┴────────────┘
                        │  results fan-in
                        ▼
                  Final Answer
```

### Key components

| Component | Description |
|---|---|
| `agent_swarm` | `@entrypoint` — the manager's tool-call agent loop |
| `spawn_sub_agent` | LangChain `@tool` bound to the manager LLM. Takes a `prompt` and makes a single LLM call |
| `run_tool_call` | LangGraph `@task` — wraps each tool invocation for parallel execution and LangSmith tracing |
| `MANAGER_SYSTEM_PROMPT` | Instructs the manager to always delegate work to sub-agents via the tool |

### How parallelism works

When the manager LLM returns multiple `tool_calls` in a single response, all are dispatched as `@task` instances before any `.result()` is awaited. This means all sub-agents run concurrently — total wall-clock time equals the slowest sub-agent, not the sum.

```
manager LLM ──► [tool_call_0, tool_call_1, ..., tool_call_N]
                      │             │                  │
                   @task          @task              @task      ← all dispatched
                      │             │                  │          before any .result()
                      └─────────────┴──────────────────┘
                                    │ fan-in
                              final answer
```

## Setup

**1. Clone and install dependencies**

```bash
cd agent-swarm
uv sync
```

**2. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `LANGSMITH_API_KEY` | API key from your [LangSmith](https://smith.langchain.com) account |
| `LANGSMITH_PROJECT` | LangSmith project name (default: `agent-swarm`) |
| `LANGSMITH_TRACING` | Set to `true` to enable tracing |

## Development

**Run**

```bash
uv run python main.py
```

**Add a dependency**

```bash
uv add <package>
```

**Update dependencies**

```bash
uv sync --upgrade
```

## Extending the swarm

The swarm is general-purpose. To adapt it to a new domain:

1. Update `MANAGER_SYSTEM_PROMPT` in [main.py](main.py) to describe the manager's role and decomposition strategy.
2. Update `EXAMPLE_QUERY` to reflect your use case.
3. Optionally add more tools alongside `spawn_sub_agent` for the manager to use.
