"""Evaluation template and the SPO-RS / SPO-GA meta-prompts."""

# Wraps the optimised system prompt and one sample into the task model's input.
EVAL_TEMPLATE = """{system_prompt}

---

**Question:**
{question}

**Context:**
{context}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your step-by-step reasoning and analysis]",
  "final_answer": "[Your concise final answer]"
}}
"""

# SPO-RS: rewrite the current best prompt, conditioned on its errors (sampled at temperature 0.9).
SPO_RS_PROMPT = """\
You are designing a system prompt for an AI that will be evaluated on the following task:

{task_description}

The current best-performing system prompt achieves {best_accuracy:.1%} accuracy:

--- Current Best Prompt ---
{best_prompt}
--- End Current Best Prompt ---

Here is a summary of its errors:

{error_summary}

Using the above prompt as a starting point, create an improved version. You should \
preserve what works but specifically target the error patterns shown above. Explore \
creative changes — restructure sections, add worked examples for the failure modes, \
switch from bullets to prose or vice versa, introduce decision trees or checklists, \
try a different role-play framing, add edge-case handling, or take a completely \
different angle. The goal is to beat {best_accuracy:.1%} accuracy.

There is no required format or structure — use whatever approach you think will be most \
effective. Experiment freely, but stay grounded in what the current best prompt does well.

The only requirement: the output must be the system prompt text itself, nothing else. No \
preamble, no explanation, no markdown fences."""

# SPO-GA crossover: combine two parents and their error patterns (temperature 0.7).
SPO_GA_CROSSOVER_PROMPT = """\
You are designing a system prompt for an AI evaluated on the following task:

{task_description}

Below are two existing system prompts for this task. Parent A scores {accuracy_a:.1%} and \
Parent B scores {accuracy_b:.1%}.

--- Parent A ---
{prompt_a}

--- Parent A error patterns ---
{error_summary_a}

--- Parent B ---
{prompt_b}

--- Parent B error patterns ---
{error_summary_b}

Create a new system prompt that combines the strengths of both parents while addressing \
their weaknesses. Prioritize strategies from the higher-scoring parent but incorporate \
any unique useful techniques from the other. Specifically target the error patterns shown \
above — if both parents fail on similar cases, introduce a new approach to handle those.

You don't need to preserve the structure of either parent — feel free to reorganize, \
reframe, or reimagine the approach entirely. What matters is that the result is effective, \
not that it resembles the inputs.

Output ONLY the new system prompt — no preamble, no explanation, no markdown fences."""

# SPO-GA mutation: targeted, error-informed rewrite of one parent (temperature 0.7).
SPO_GA_MUTATION_PROMPT = """\
Below is a system prompt used for the following task:

{task_description}

It currently achieves {accuracy:.1%} accuracy. Here is a summary of its errors:

{error_summary}

First, analyze the error patterns above — look for common failure modes such as wrong \
scale or units, misinterpreting instructions, incorrect formula application, rounding \
errors, off-by-one mistakes, missing edge cases, or format mismatches. Then produce an \
improved version that specifically targets these failure patterns.

You may restructure it however you like — change the framing, add worked examples for \
the failure modes you identified, switch from bullets to prose, introduce decision trees \
or checklists, or take a completely different angle. The goal is to fix the errors; how \
you get there is up to you.

Output ONLY the improved system prompt — no preamble, no explanation, no markdown fences.

--- Current Prompt ---
{prompt}"""
