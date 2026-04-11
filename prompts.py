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
""")

BROWSER_USE_QUERY = textwrap.dedent("""\
    Please find my user name on X (formerly Twitter). I have already logged in.
""")
