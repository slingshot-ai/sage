"""System prompts for the four SAGE agents."""

from .analyzer import ANALYZER_PROMPT
from .generator import GENERATOR_PROMPT
from .investigator import INVESTIGATOR_PROMPT
from .orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

__all__ = ["ANALYZER_PROMPT", "GENERATOR_PROMPT", "INVESTIGATOR_PROMPT", "ORCHESTRATOR_SYSTEM_PROMPT"]
