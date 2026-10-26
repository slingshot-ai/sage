"""Per-task prompts: the seed prompt, the SPO-RS/SPO-GA task description, and the SAGE agent context."""

from types import ModuleType

from . import appworld, finer, formula

TASKS: dict[str, ModuleType] = {"finer": finer, "formula": formula, "appworld": appworld}
