"""Proposal strategies, keyed by their ``--strategy`` name."""

from .agentic import SAGE
from .base import Strategy
from .genetic import GeneticAlgorithm
from .residual_search import ResidualSearch

STRATEGIES: dict[str, type[Strategy]] = {"spo-rs": ResidualSearch, "spo-ga": GeneticAlgorithm, "sage": SAGE}
