"""
Agent Swarm — LangGraph Functional API + Tool-Calling Manager
Architecture:
  - `spawn_sub_agent` is a LangChain tool bound to the manager LLM
  - The manager LLM autonomously decides how many sub-agents to spawn and what
    to assign each one, based on the user's instructions
  - All sub-agent calls are dispatched in parallel via LangGraph @task
  - Manager runs a tool-call agent loop until it produces a final answer
  - LangSmith traces every LLM call and tool invocation
  - `browse_web` uses browser-use + Playwright; Gemini drives all browser
    decisions from raw page state (no browser-use cloud API key needed)
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from browser_use import Agent as BrowserAgent, BrowserProfile, ChatGoogle
from browser_use.browser.profile import BrowserChannel
from browser_use.browser.views import BrowserStateSummary
from browser_use.agent.views import AgentOutput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.func import entrypoint, task
import prompts

load_dotenv()

RECORDINGS_DIR = Path("recordings")
CHROME_USER_DATA_DIR = Path.home() / "Library/Application Support/Google/Chrome"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3-flash-preview"


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Tool: browse_web
# Drives a real browser via Playwright. The Gemini LLM receives raw page
# state (HTML structure + screenshot) and decides every browser action —
# no browser-use API key is used.
# Browser runs headed (visible window) so you can watch every click/scroll
# live. A demo panel is overlaid inside the browser showing agent thoughts,
# and each step is also printed to the terminal in real time.
# ---------------------------------------------------------------------------

def _on_browser_step(browser_state: BrowserStateSummary, agent_output: AgentOutput, step_n: int) -> None:
    """Print live step info to the terminal as the browser agent operates."""
    print(
        f"\n[Browser Step {step_n}] {browser_state.title} — {browser_state.url}")
    if agent_output.next_goal:
        print(f"  Goal : {agent_output.next_goal}")
    for action in agent_output.action:
        print(f"  Act  : {action.model_dump(exclude_none=True, mode='json')}")


@tool
def browse_web(task: str) -> str:
    """
    Browse the web to complete a task using browser automation.
    Include any relevant URLs directly in the task description.
    Example: 'Go to https://example.com and find the current pricing plans'
    Returns the final extracted result from the browsing session.
    Each run is recorded as an MP4 video, a GIF, and a JSON action log
    saved under recordings/<timestamp>/.
    """
    async def _run() -> str:
        run_dir = RECORDINGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        agent = BrowserAgent(
            task=task,
            llm=ChatGoogle(
                model=GEMINI_MODEL, api_key=os.environ["GOOGLE_API_KEY"], temperature=0.2),
            browser_profile=BrowserProfile(
                headless=False,
                channel=BrowserChannel.CHROME,
                user_data_dir=CHROME_USER_DATA_DIR,
                record_video_dir=run_dir,
            ),
            register_new_step_callback=_on_browser_step,
            demo_mode=True,
            generate_gif=str(run_dir / "session.gif"),
        )
        history = await agent.run()
        history.save_to_file(run_dir / "actions.json")
        print(f"\n[Recording saved] {run_dir.resolve()}")
        return history.final_result() or "Task completed but no text result was extracted."

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tool: spawn_sub_agent
# Provided to the manager LLM. Sub-agents run their own tool loop so they
# can also call browse_web when needed.
# ---------------------------------------------------------------------------

SUB_AGENT_TOOLS = [browse_web]

@tool
def spawn_sub_agent(prompt: str) -> str:
    """
    Spawn a sub-agent to handle a single task.
    Provide a complete, self-contained prompt. The sub-agent has access to
    the browse_web tool for any web-related work and will respond with the
    result of that task.
    """
    sub_llm = get_llm().bind_tools(SUB_AGENT_TOOLS)
    sub_tools_by_name: dict = {t.name: t for t in SUB_AGENT_TOOLS}

    messages: list[BaseMessage] = [HumanMessage(content=prompt)]

    while True:
        response: AIMessage = sub_llm.invoke(
            messages)  # type: ignore[assignment]
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            result = str(
                sub_tools_by_name[tool_call["name"]].invoke(tool_call["args"]))
            messages.append(ToolMessage(
                content=result, tool_call_id=tool_call["id"]))

    return str(response.content).strip()


# ---------------------------------------------------------------------------
# Entrypoint: manager agent loop
# ---------------------------------------------------------------------------

@entrypoint()
def agent_swarm(user_query: str) -> str:
    tools = [browse_web]
    manager_llm = get_llm().bind_tools(tools)
    tools_by_name: dict = {t.name: t for t in tools}

    messages: list[BaseMessage] = [
        SystemMessage(content=prompts.BROWSER_USE_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]

    # Agent loop: keep going until the manager produces a response with no tool calls
    while True:
        response: AIMessage = manager_llm.invoke(messages)  # type: ignore[assignment]
        messages.append(response)

        if not response.tool_calls:
            # Manager is done — final text answer
            break

        # Dispatch all tool calls as parallel @task instances
        @task
        def run_tool_call(tool_name: str, tool_args: dict) -> str:
            return str(tools_by_name[tool_name].invoke(tool_args))

        futures = [
            (tool_call["id"], run_tool_call(tool_call["name"], tool_call["args"]))
            for tool_call in response.tool_calls
        ]

        # Fan-in: collect all results (blocks until every sub-agent finishes)
        for tool_call_id, future in futures:
            tool_result = future.result()
            print(f"  [Sub-agent] returned: {tool_result[:80]}...")
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                )
            )

    return str(response.content).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=== Agent Swarm ===")
    print("LangSmith tracing:", os.getenv("LANGSMITH_TRACING", "false"))

    print("\nRunning swarm...\n")
    result = agent_swarm.invoke(prompts.BROWSER_USE_QUERY_GET_TWEETS)

    print("\n--- Final Answer ---")
    print(result)
    print("\n=== Done ===")

    langsmith_project = os.getenv("LANGSMITH_PROJECT", "agent-swarm")
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        print(f"\nView trajectory in LangSmith project: {langsmith_project}")


if __name__ == "__main__":
    main()
