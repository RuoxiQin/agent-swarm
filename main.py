"""
Agent Swarm — LangGraph Functional API + Tool-Calling Manager
Architecture:
  - `spawn_sub_agent` is a LangChain tool bound to the manager LLM
  - The manager LLM autonomously decides how many sub-agents to spawn and what
    to assign each one, based on the user's instructions
  - All sub-agent calls are dispatched in parallel via LangGraph @task
  - Manager runs a tool-call agent loop until it produces a final answer
  - LangSmith traces every LLM call and tool invocation
"""

import os
from dotenv import load_dotenv
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
from prompts import EXAMPLE_QUERY, MANAGER_SYSTEM_PROMPT

load_dotenv()

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
# Tool: spawn_sub_agent
# Provided to the manager LLM. The manager decides when and how many times
# to call it, and what prompt to give each sub-agent.
# ---------------------------------------------------------------------------

@tool
def spawn_sub_agent(prompt: str) -> str:
    """
    Spawn a sub-agent to handle a single task.
    Provide a complete, self-contained prompt. The sub-agent will respond with
    the result of that task.
    """
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()


# ---------------------------------------------------------------------------
# Entrypoint: manager agent loop
# ---------------------------------------------------------------------------

@entrypoint()
def agent_swarm(user_query: str) -> str:
    tools = [spawn_sub_agent]
    manager_llm = get_llm().bind_tools(tools)
    tools_by_name: dict = {t.name: t for t in tools}

    messages: list[BaseMessage] = [
        SystemMessage(content=MANAGER_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]

    # Agent loop: keep going until the manager produces a response with no tool calls
    while True:
        response: AIMessage = manager_llm.invoke(messages)  # type: ignore[assignment]
        messages.append(response)

        if not response.tool_calls:
            # Manager is done — final text answer
            break

        print(f"[Manager] Dispatching {len(response.tool_calls)} sub-agent(s) in parallel...")

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
    result = agent_swarm.invoke(EXAMPLE_QUERY)

    print("\n--- Final Answer ---")
    print(result)
    print("\n=== Done ===")

    langsmith_project = os.getenv("LANGSMITH_PROJECT", "agent-swarm")
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        print(f"\nView trajectory in LangSmith project: {langsmith_project}")


if __name__ == "__main__":
    main()
