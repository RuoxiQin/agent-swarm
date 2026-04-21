import ast
import html
import json
import os
import re
import textwrap
import time
import urllib.request
from datetime import datetime
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
        elif event_type == "tool_execution_end":
            print(f"\n[tool result: {data}]")
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


def _parse_session_analysis(response_str: str) -> dict:
    """Extract tokens, cost, and the last-message text from a str(PiAgentSessionAnalysis)."""
    if not isinstance(response_str, str):
        return {"ok": False, "tokens": None, "cost": None, "text": ""}
    tokens_m = re.search(r"total_tokens=(\d+)", response_str)
    cost_m = re.search(r"total_cost=([\d.eE+\-]+)", response_str)
    text_m = re.search(
        r"text='(.*)',\s*timestamp=\d+\)\)\s*$", response_str, flags=re.DOTALL
    )
    text = ""
    if text_m:
        raw = text_m.group(1)
        try:
            text = ast.literal_eval("'" + raw + "'")
        except (ValueError, SyntaxError):
            text = raw
    return {
        "ok": bool(tokens_m and text_m),
        "tokens": int(tokens_m.group(1)) if tokens_m else None,
        "cost": float(cost_m.group(1)) if cost_m else None,
        "text": text,
    }


def render_html(json_path: str, output_path: str) -> None:
    with open(json_path, encoding="utf-8") as f:
        per_example = json.load(f)

    esc = html.escape

    def render_rank_badge(rank_index: int | None, is_tied: bool) -> str:
        if rank_index is None:
            badge = "unranked"
            cls = "badge badge-unranked"
        else:
            badge = f"#{rank_index + 1}" + (" (tied)" if is_tied else "")
            cls = f"badge badge-rank-{min(rank_index + 1, 3)}"
        return f'<span class="{cls}">{esc(badge)}</span>'

    examples_html = []
    for item in per_example:
        example_id = item["example_id"]
        evaluation = item.get("evaluation") or {}
        results = item.get("results") or {}
        ranking = evaluation.get("ranking") or []
        tied = evaluation.get("tied") or []
        verdicts = evaluation.get("verdicts") or {}
        reason = evaluation.get("reason") or ""

        rank_index_by_label = {label: i for i, label in enumerate(ranking)}
        tied_flat = [label for group in tied for label in (
            group if isinstance(group, list) else [group])]
        tied_set = set(tied_flat)

        if ranking:
            top = ranking[0]
            best_labels = [top]
            for group in tied:
                group_list = group if isinstance(group, list) else [group]
                if top in group_list:
                    best_labels = list(dict.fromkeys([top, *group_list]))
                    break
            best_html = ", ".join(
                f'<span class="rank-pill">{esc(label)}</span>' for label in best_labels
            )
            if len(best_labels) > 1:
                best_html += ' <span class="muted">(tied)</span>'
        else:
            best_html = '<span class="muted">(no ranking)</span>'
        tied_html = (
            "; ".join(
                ", ".join(esc(t)
                          for t in (group if isinstance(group, list) else [group]))
                for group in tied
            )
            if tied
            else '<span class="muted">none</span>'
        )

        metrics_rows = []
        si_cards = []
        unranked_sentinel = len(ranking) + 1
        ordered_items = sorted(
            results.items(),
            key=lambda kv: (
                rank_index_by_label.get(kv[0], unranked_sentinel),
                kv[0],
            ),
        )
        for si_key, entry in ordered_items:
            si_content = entry.get("si", "")
            response = entry.get("response", "")
            verdict = verdicts.get(si_key, "")
            rank_index = rank_index_by_label.get(si_key)
            is_tied = si_key in tied_set

            parsed = _parse_session_analysis(response)
            elapsed = entry.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                time_str = f"{elapsed:.1f}s"
            else:
                time_str = '<span class="muted">n/a</span>'
            result_str = (
                '<span class="result-ok">succeeded</span>'
                if parsed["ok"]
                else '<span class="result-fail">failed</span>'
            )
            tokens_str = (
                f"{parsed['tokens']:,}"
                if parsed["tokens"] is not None
                else '<span class="muted">n/a</span>'
            )
            cost_str = (
                f"${parsed['cost']:.4f}"
                if parsed["cost"] is not None
                else '<span class="muted">n/a</span>'
            )
            rank_cell = render_rank_badge(rank_index, is_tied)
            metrics_rows.append(
                f"""<tr>
              <td class="metric-approach">{esc(si_key)}</td>
              <td>{rank_cell}</td>
              <td>{result_str}</td>
              <td>{time_str}</td>
              <td>{tokens_str}</td>
              <td>{cost_str}</td>
            </tr>"""
            )

            response_text = parsed["text"] if parsed["text"] else response
            si_cards.append(
                f"""
        <div class="si-card">
          <div class="si-card-header">
            <h3>{esc(si_key)}</h3>
            {render_rank_badge(rank_index, is_tied)}
          </div>
          <div class="field">
            <div class="field-label">System instruction</div>
            <pre class="field-value">{esc(si_content)}</pre>
          </div>
          <div class="field">
            <div class="field-label">Response</div>
            <pre class="field-value response">{esc(response_text)}</pre>
          </div>
          <div class="field">
            <div class="field-label">Verdict</div>
            <div class="field-value verdict">{esc(verdict) if verdict else '<span class="muted">(none)</span>'}</div>
          </div>
        </div>"""
            )

        metrics_html = f"""
        <table class="metrics-table">
          <thead>
            <tr><th>Approach</th><th>Rank</th><th>Result</th><th>Time</th><th>Tokens</th><th>Cost</th></tr>
          </thead>
          <tbody>
            {''.join(metrics_rows)}
          </tbody>
        </table>"""

        examples_html.append(
            f"""
    <section class="example">
      <header class="example-header">
        <h2>{esc(example_id)}</h2>
      </header>
      <div class="summary">
        <div class="summary-row"><span class="summary-label">Best</span><span class="summary-value">{best_html}</span></div>
        <div class="summary-row"><span class="summary-label">Tied</span><span class="summary-value">{tied_html}</span></div>
        <div class="summary-row"><span class="summary-label">Reason</span><span class="summary-value">{esc(reason) if reason else '<span class="muted">(none)</span>'}</span></div>
      </div>
      {metrics_html}
      <div class="si-grid">
        {''.join(si_cards)}
      </div>
    </section>"""
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Eval Results</title>
<style>
  :root {{
    --bg: #f7f7f8;
    --surface: #ffffff;
    --border: #e5e5ea;
    --text: #1d1d1f;
    --muted: #86868b;
    --accent: #0071e3;
    --accent-soft: #e8f1fc;
    --code-bg: #f4f4f5;
    --rank-1: #1a7f37;
    --rank-2: #9a6700;
    --rank-3: #8250df;
    --rank-other: #57606a;
    --rank-unranked: #6e7781;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 32px 24px 80px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  .page {{ max-width: 1100px; margin: 0 auto; }}
  .page-header {{ margin-bottom: 32px; }}
  .page-header h1 {{ margin: 0 0 4px; font-size: 28px; font-weight: 600; }}
  .page-header .meta {{ color: var(--muted); font-size: 13px; }}
  .example {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .example-header h2 {{
    margin: 0 0 16px;
    font-size: 18px;
    font-weight: 600;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text);
  }}
  .summary {{
    background: var(--accent-soft);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 20px;
    font-size: 14px;
  }}
  .summary-row {{ display: flex; gap: 12px; padding: 3px 0; }}
  .summary-label {{
    flex: 0 0 80px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.04em;
    padding-top: 2px;
  }}
  .summary-value {{ flex: 1; word-break: break-word; }}
  .rank-pill {{
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px 10px;
    margin: 0 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
  }}
  .si-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 16px;
  }}
  .si-card {{
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    background: var(--surface);
  }}
  .si-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }}
  .si-card-header h3 {{
    margin: 0;
    font-size: 15px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    background: var(--rank-other);
  }}
  .badge-rank-1 {{ background: var(--rank-1); }}
  .badge-rank-2 {{ background: var(--rank-2); }}
  .badge-rank-3 {{ background: var(--rank-3); }}
  .badge-unranked {{ background: var(--rank-unranked); }}
  .field {{ margin-top: 10px; }}
  .field-label {{
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }}
  .field-value {{ font-size: 13px; }}
  pre.field-value {{
    margin: 0;
    padding: 10px 12px;
    background: var(--code-bg);
    border-radius: 6px;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12.5px;
    max-height: 320px;
    overflow: auto;
  }}
  pre.response {{ max-height: 420px; }}
  .verdict {{
    padding: 8px 12px;
    background: #fff8e6;
    border-left: 3px solid #f0b429;
    border-radius: 4px;
    font-size: 13px;
  }}
  .muted {{ color: var(--muted); font-style: italic; }}
  .metrics-table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    font-size: 13px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .metrics-table th, .metrics-table td {{
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }}
  .metrics-table tr:last-child td {{ border-bottom: none; }}
  .metrics-table th {{
    background: var(--code-bg);
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.04em;
  }}
  .metric-approach {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600;
  }}
  .result-ok {{ color: var(--rank-1); font-weight: 600; }}
  .result-fail {{ color: #cf222e; font-weight: 600; }}
</style>
</head>
<body>
  <div class="page">
    <header class="page-header">
      <h1>Eval Results</h1>
      <div class="meta">Generated {esc(generated_at)} &middot; {len(per_example)} example(s)</div>
    </header>
    {''.join(examples_html)}
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)


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
            started_at = time.time()
            agent_response: str = await ask_agent(si_content, query)
            elapsed_seconds = time.time() - started_at
            results[si_key] = {
                "si": si_content,
                "response": agent_response,
                "elapsed_seconds": elapsed_seconds,
            }

        print("Comparing all candidate results...")
        evaluation = evaluate_agents(
            query,
            golden_answer,
            [
                (
                    si_key,
                    f"elapsed_seconds: {entry['elapsed_seconds']:.2f}\n{entry['response']}",
                )
                for si_key, entry in results.items()
            ],
        )
        per_example.append({
            "example_id": example_id,
            "evaluation": evaluation,
            "results": results,
        })

    out_dir = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"eval_results_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(per_example, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON results written to: {json_path}")

    html_path = os.path.join(out_dir, f"eval_results_{stamp}.html")
    render_html(json_path, html_path)
    print(f"HTML report written to: {html_path}")

    return per_example
