import json
import os
import textwrap
import urllib.request
from dotenv import load_dotenv
from funky import PiAgentWorkspace
from funky.errors import APIError
import prompts

load_dotenv()

GEMINI_MODEL = "gemini-3-flash-preview"


async def ask_agent(si: str, query: str) -> str:
    ws = PiAgentWorkspace.create(api_key=os.environ["FUNKY_API_KEY"])
    print(f"workspace ready: {ws.claim_name} (pod={ws.pod_name})")
    session = ws.create_session(model="gemini-3-flash-preview")
    print(f"session: {session.session_id} model={session.model}")

    for event_type, data in ws.send_message(
        session.session_id,
        si + "\n" + query,
        timeout=300.0,  # SSE streams can be long; override the default
    ):
        if event_type == "message_update":
            evt = data.get("assistantMessageEvent", {})
            if evt.get("type") == "text_delta":
                print(evt.get("delta", ""), end="", flush=True)
        elif event_type == "tool_execution_start":
            print(f"\n[tool call: {data}]")
        elif event_type == "agent_end":
            print("\n[done]")
            break
        elif event_type == "error":
            print(f"\n[error] {data}")
            break

    results = str(ws.get_session_analysis(session.session_id))
    await ws.delete()

    return results

def evaluate_agents(
    task_description: str,
    golden_answer: str,
    agent_responses: list[tuple[str, str]],
) -> dict:
    labels = [label for label, _ in agent_responses]
    if len(set(labels)) != len(labels):
        raise ValueError(f"agent labels must be unique, got: {labels}")

    responses_block = "\n\n".join(
        f"[{label}_RESPONSE_BEGIN]\n{response}\n[{label}_RESPONSE_END]"
        for label, response in agent_responses
    )
    query = prompts.AUTORATER_QUERY.format(
        task_description=task_description,
        golden_answer=golden_answer,
        agent_responses=responses_block,
        agent_labels=", ".join(labels),
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
        f":generateContent?key={os.environ['GOOGLE_API_KEY']}"
    )
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": prompts.AUTORATER_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": query}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])


AGENT_1_RESPONSE = """{
  "id": "01353c30-7ba5-4721-a3ef-563e7948ced0",
  "totalTokens": 849517,
  "totalCost": 0.13416105000000003,
  "lastMessage": {
    "role": "assistant",
    "text": "Based on my search on the Kirkwood website for April 18 to April 19, 2026, for one adult:\n\n*   The cheapest lodge is the Meadows Region.\n*   The cheapest room at this lodge with at least one queen bed is the Meadows Region | Studio, which costs $293 (Online rate).\n*   While there is a "Hotel Room" option in the same region for $258, it only features one double bed. The Studio unit specifically features one queen bed, a full bath, a kitchenette, and a wood-burning fireplace.\n\nOther lodges like The Mountain Club (starting at $392) and Sun Meadows (starting at $436) are significantly more expensive for these dates.",
    "timestamp": 1776449291810
  }
}"""

AGENT_2_RESPONSE = """{
  "id": "1dc4b89a-54f7-44d7-96c1-b2750912447d",
  "totalTokens": 461006,
  "totalCost": 0.07758155,
  "lastMessage": {
    "role": "assistant",
    "text": "Based on my search for Kirkwood lodging for the night of April 18 to April 19, 2026, the cheapest room with at least one queen bed is:\n\n*   Lodge: Meadows Region\n*   Room Type: Studio\n*   Price: $293 (Nightly, including fees)\n*   Bed Configuration: 1 Queen Bed\n\nWhile there is a slightly cheaper "Hotel Room" option in the Meadows Region for $258, it only features one double bed. The Studio is the most affordable option that meets your requirement of having at least one queen bed. \n\n(Note: If you have an Epic Pass, you may be eligible for a discounted "Rewards" rate of approximately $234 for this room.)",
    "timestamp": 1776451221476
  }
}"""

BROWSER_USE_QUERY_BOOK_LODGE_UPDATED = textwrap.dedent("""\
    Please use `playwright-cli --help` to learn how to browse web.
    I want to book a lodge in Kirkwood for 4/18/2026 to 4/19/2026 (1 night, 1 person). Please help me check the cheapest room with at least 1 queen bed. Please exclude the room you're not confident to have at least one queen size bed (e.g. having conflicting bed types in the description). Their website is https://www.kirkwood.com/plan-your-trip/stay/kirkwood-lodging.aspx. Which lodge is it and how much?
""")


async def evaluate_candidates(system_instructions_to_evaluate, evals_json_path):
    with open(evals_json_path) as f:
        examples = json.load(f)["examples"]

    per_example = []
    for example in examples:
        example_id = example["example_id"]
        query = example["query"]
        golden_answer = example["golden_answer"]

        print(f"\n=== Example: {example_id} ===")
        results = {}
        for si_key, si_content in system_instructions_to_evaluate.items():
            print(f"Evaluating {si_key}...")
            agent_response: str = await ask_agent(si_content, query)
            results[si_key] = {"si": si_content, "response": agent_response}

        print("Comparing all candidate results...")
        evaluation = evaluate_agents(
            query,
            golden_answer,
            [(si_key, entry["response"]) for si_key, entry in results.items()],
        )
        per_example.append({
            "example_id": example_id,
            "evaluation": evaluation,
            "results": results,
        })

    return per_example
