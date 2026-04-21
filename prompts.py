import textwrap

# Prompts for parallel translations.
MANAGER_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a manager agent. You have access to the `spawn_sub_agent` tool.

    When given a task that can be broken into parallel sub-tasks:
    1. Identify the individual sub-tasks.
    2. Spawn one sub-agent per sub-task by calling `spawn_sub_agent` with a
       complete, self-contained prompt for that sub-task.
    3. Once all sub-agents have responded, compile their results and return a
       final answer to the user.

    You don't have to delegat tasks. If the task won't benefit from parallelization (e.g. step-by-step research) or you can handle by yourself, then just do the work yourself.
""")

EXAMPLE_QUERY = textwrap.dedent("""\
    Please translate the following sentences to Chinese.
    Spawn the same number of sub-agents as there are sentences
    and assign one sentence per sub-agent:

    1. The weather today is beautiful and sunny.
    2. I would like a cup of coffee, please.
    3. Where is the nearest train station?
    4. Learning a new language opens many doors.
    5. Thank you very much for your help.
""")

# Prompts for browser use.
BROWSER_USE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an agent with access to a web browser tool.
    Once you have obtained the requested information, return your answer immediately.
    Do not browse again to verify or confirm — one successful result is enough to stop.
""")

BROWSER_USE_QUERY_GET_USERNAME = textwrap.dedent("""\
    Please find my user name on X (formerly Twitter). I have already logged in.
""")

BROWSER_USE_QUERY_GET_TWEETS = textwrap.dedent("""\
    Please open X (formally Twitter), and find the latest 5 tweets by my friend `@XiJin12`. Find the post text content, their number of views, and number of likes.
""")

BROWSER_USE_QUERY_BOOK_LODGE = textwrap.dedent("""\
    I want to book a lodge in Kirkwood for 4/18/2026 to 4/19/2026 (1 night, 1 person). Please help me check the cheapest room with at least 1 queen bed. Their website is https://www.kirkwood.com/plan-your-trip/stay/kirkwood-lodging.aspx. Which lodge is it and how much?
    Note that after you input the date, you'll need to click at a blank place to confirm the input.
""")

BROWSER_USE_QUERY_BOOK_LODGE_GOLDEN_ANSWER = textwrap.dedent("""\
    The cheapest room with a queen bed is **Meadows Region | Studio** at $293. If you have Epic pass, the reward price is $234. Note that although **Meadows Region | Hotel Room** is $258, its description is conflicting on the bed size so it's not recommended.
""")

# Autorater agent

AUTORATER_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a rater agent. You will be given a task description, a golden answer, and the outputs from one or more agents identified by label. Your task is to evaluate each agent's response against the golden answer and rank them from best to worst, with reasons.
""")

AUTORATER_QUERY = textwrap.dedent("""\
    Please evaluate the agent responses below against the golden answer for the given task. Rank the responses from best to worst and explain why.

    [TASK_DESCRIPTION_BEGIN]
    {task_description}
    [TASK_DESCRIPTION_END]

    [GOLDEN_ANSWER_BEGIN]
    {golden_answer}
    [GOLDEN_ANSWER_END]

    [AGENT_RESPONSES_BEGIN]
    {agent_responses}
    [AGENT_RESPONSES_END]

    The agent labels to rank are: {agent_labels}.

    Compare each agent's response to the golden answer on correctness, completeness, and clarity. Reply with a JSON object and nothing else:
    {{"ranking": ["<label>", ...], "tied": [["<label>", "<label>"], ...], "verdicts": {{"<label>": "<one-line verdict per agent>"}}, "reason": "<concise overall justification>"}}

    "ranking" must be a permutation of the provided agent labels from best to worst. Put agents you consider equivalent next to each other in the ranking and list each tie-group in "tied" (omit or leave empty if there are no ties).
""")
