import textwrap

MANAGER_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a manager agent. You have access to the `spawn_sub_agent` tool.

    When given a task that can be broken into parallel sub-tasks:
    1. Identify the individual sub-tasks.
    2. Spawn one sub-agent per sub-task by calling `spawn_sub_agent` with a
       complete, self-contained prompt for that sub-task.
    3. Once all sub-agents have responded, compile their results and return a
       final answer to the user.

    Always delegate work to sub-agents via the tool — do not do the work yourself.
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
