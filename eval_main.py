import asyncio

from eval_utils import evaluate_candidates

system_instructions_to_evaluate = {
    "agent-brwoser": "Please use `agent-browser --help` to learn how to browse web.",
    "playwright-cli": "Please use `playwright-cli --help` to learn how to browse web."
}


async def main():
    per_example = await evaluate_candidates(
        system_instructions_to_evaluate,
        "lodge_booking_evals.json",
    )
    for item in per_example:
        evaluation = item["evaluation"]
        print(f"\n--- {item['example_id']} ---")
        print(f"Per-SI results: {item['results']}")
        print(f"Ranking : {evaluation.get('ranking')}")
        print(f"Tied    : {evaluation.get('tied', [])}")
        print(f"Verdicts: {evaluation.get('verdicts')}")
        print(f"Reason  : {evaluation.get('reason')}")


if __name__ == "__main__":
    asyncio.run(main())
