"""Prompts for AppWorld: interactive API coding agent (the optimised prompt is the agent's playbook)."""

# Initial prompt that every strategy starts from.
SEED_PROMPT = """\
Make sure to end code blocks with ``` followed by a newline.

Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API. Write small chunks of code — only one code block per step. Make sure everything works correctly before making any irreversible change.

Use the "supervisor" app to get information about your accounts and the "phone" app to get information about friends and family.

Many APIs return items in "pages". Always loop over `page_index` to get all results.

Variables you define in one code block carry over to subsequent blocks. However, email addresses, access tokens, and credentials from the few-shot examples above are not valid — always retrieve fresh ones.

When the task is complete, call apis.supervisor.complete_task(). If the task asks for information, pass it as the answer argument: apis.supervisor.complete_task(answer=...). Many tasks do not require an answer — in those cases call complete_task() with no arguments."""

# Task description shown to the SPO-RS and SPO-GA meta-model.
DESCRIPTION = """\
AppWorld Agent Task (Interactive API Coding)

Complete day-to-day tasks autonomously by writing Python code that interacts with app APIs (e.g., Spotify, Venmo, phone, email) in a simulated digital environment.

The agent operates in a ReAct loop: think, write code, observe output, repeat. The agent receives a 'playbook' — guidance text covering strategies, API tips, common mistakes, and proven solutions — as part of its context. The playbook can be any format: prose, bullet points, sections, decision trees, code snippets, or any mix.

Input: A natural-language task instruction (e.g., 'How many playlists do I have in Spotify?') plus access to app APIs.

Output: The agent must complete the task by calling the correct APIs in the right sequence, then call apis.supervisor.complete_task() with the answer.

Evaluation: Task Goal Completion (TGC) — binary per task, all unit test assertions must pass. Scenario Goal Completion (SGC) — fraction of assertions that pass across all tasks.

Key challenges: API discovery and pagination, multi-app coordination, credential management, parsing unstructured content, edge cases in data formats."""

# Richer task description shown to the SAGE agents.
AGENT_DESCRIPTION = """\
AppWorld Agent Task (Interactive API Coding)

Complete day-to-day tasks autonomously by writing Python code that interacts with app APIs (e.g., Spotify, Venmo, phone, email) in a simulated digital environment.

The agent operates in a ReAct loop: think, write code, observe output, repeat. The agent's context includes a 'playbook' — guidance text that shapes its problem-solving approach. The playbook format is flexible (prose, bullets, sections, decision trees, code snippets, etc.).

**Input format:** Each task is a natural-language instruction, e.g., 'How many playlists do I have in Spotify?' or 'Send a text to Christopher with my movie recommendations.'

**Output format:** The agent writes Python code that calls APIs and eventually calls `apis.supervisor.complete_task(answer=...)` when done.

**Evaluation:** Task Goal Completion (TGC) — binary per task, all unit-test assertions must pass. Scenario Goal Completion (SGC) — fraction of assertions that pass.

**What is being optimized:** The 'playbook' text — guidance injected into the agent's prompt. It can use any format or structure. The content typically covers strategies, API tips, code snippets, common mistakes, heuristics, and troubleshooting — but the format is entirely up to the optimizer. Better playbooks → higher TGC/SGC.

**Key challenges:** API discovery and pagination, multi-app coordination, credential management, parsing unstructured content, edge cases in data formats, correctly formatting answers (e.g., comma-separated vs single value)."""

# Task-specific error-analysis guidance for the SAGE agents.
ANALYSIS_GUIDANCE = """\
## Task-Specific Analysis Guidance (AppWorld / Interactive Agent Tasks)

This is an interactive coding agent benchmark. The agent writes Python code to interact with simulated app APIs. The "prompt" being optimized is a *playbook* that guides the agent's behavior. The playbook format is completely flexible — prose, bullet points, numbered lists, decision trees, code snippets, or any combination. Experiment freely with structure.

**High-value analysis patterns:**
- **Failure category analysis:** Group failed tasks by the type of failure: wrong API used, pagination not handled, incorrect answer format, missing credential retrieval, wrong data parsing, timeout/step limit exceeded.
- **App-level analysis:** Check which apps (spotify, venmo, phone, email, etc.) appear most in failures. Some apps may have tricky pagination or non-obvious API patterns.
- **Multi-step coordination failures:** Many tasks require coordinating across 2-3 apps (e.g., find contact via phone, then send via email). Check if failures cluster at coordination points.
- **Answer format errors:** The agent may complete the task logic correctly but format the answer wrong in `complete_task(answer=...)`. Check if the assertion failures mention specific expected values.

**What makes an effective AppWorld playbook fix:**
- Concrete API usage patterns: "When searching for contacts, always use `search_contacts` with the person's first name, then verify the result before proceeding."
- Pagination rules: "Always use `while True` loops for pagination instead of fixed ranges."
- Answer formatting rules: "For count questions, pass the integer directly; for name questions, pass the exact string."
- Common pitfalls: "Some APIs return nested objects; always check the response structure before accessing fields."
- Multi-app coordination patterns: "To send a message, first get credentials from supervisor, then login to the app, then send."
- Verification steps: "Before calling complete_task, verify your answer by re-checking the relevant data.\""""
