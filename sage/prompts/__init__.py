"""All prompts used by SPO: evaluation template, meta-prompts, SAGE agent prompts and per-task prompts."""

from .tasks import TASKS
from .templates import EVAL_TEMPLATE, SPO_GA_CROSSOVER_PROMPT, SPO_GA_MUTATION_PROMPT, SPO_RS_PROMPT

__all__ = ["EVAL_TEMPLATE", "SPO_GA_CROSSOVER_PROMPT", "SPO_GA_MUTATION_PROMPT", "SPO_RS_PROMPT", "TASKS"]
