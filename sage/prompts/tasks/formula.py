"""Prompts for XBRL Formula: financial formula calculation."""

# Initial prompt that every strategy starts from.
SEED_PROMPT = """\
You are an expert financial calculator with deep knowledge of accounting, finance, and quantitative analysis.

Your task is to solve financial calculation problems accurately and step by step.

Guidelines:
- Read the question carefully and identify what is being asked
- Break down complex calculations into clear, sequential steps
- Apply the correct financial formulas and relationships
- Perform necessary unit conversions (e.g., "5 million" = 5000000.0, percentages to decimals)
- Double-check arithmetic at each step
- Round your final answer to the nearest hundredth when necessary
- Your final answer should be a plain floating point number"""

# Task description shown to the SPO-RS and SPO-GA meta-model.
DESCRIPTION = """\
Financial Formula Calculation

Solve quantitative finance problems requiring arithmetic, formula application, unit conversions, or multi-step calculations.

Input: A financial question, optionally with tables or context.

Output: A single floating-point number rounded to the nearest hundredth.

Key challenges: Output scale conventions (decimal vs percentage), unit conversion (millions/billions to base units), intermediate precision."""

# Richer task description shown to the SAGE agents.
AGENT_DESCRIPTION = """\
Financial Formula Calculation

Solve quantitative finance problems that require arithmetic, formula application, unit conversions, or multi-step calculations.

**Input format:** A financial question (e.g., 'What is the current ratio given current assets of $60,000 and current liabilities of $30,000?'), optionally with additional context (tables, financial data).

**Output format:** A single floating-point number rounded to the nearest hundredth. Example: '2.0', '0.05', '640000000000.0'

**Key challenges:**
  - **Output scale conventions:** The benchmark has specific conventions for whether a metric should be expressed as a decimal or a percentage. For instance, WACC of 5% should be output as 0.05 (decimal), while ROE of 10% should be output as 10.0 (percentage). Confusing these conventions is historically the single largest error category (~30-40% of all errors).
  - **Unit conversion:** Values stated in millions or billions must be converted to base units in the final answer (e.g., '3 billion' = 3000000000.0).
  - **Intermediate precision:** Rounding intermediate values too early causes cascading errors. Full precision must be carried until the final answer.
  - **Formula conventions:** Some formulas have multiple conventions (e.g., ordinary annuity vs annuity-due, discrete vs continuous compounding).

**Correctness:** The predicted number must exactly match the target after both are parsed as floats (with commas stripped)."""

# Task-specific error-analysis guidance for the SAGE agents.
ANALYSIS_GUIDANCE = """\
## Task-Specific Analysis Guidance (Formula / Financial Calculation)

This is a numerical computation task where errors fall into distinct, classifiable categories.

**High-value analysis patterns:**
- **Error categorization:** Run `error_analysis.py --scale` to see the breakdown of error types (percentage vs decimal, unit scaling, rounding, large errors). This immediately reveals which error category dominates.
- **Predicted/target ratio analysis:** For each error, compute predicted/target. Ratios near 100 or 0.01 indicate scale convention errors; ratios near 1000 indicate unit errors; ratios near 1.0 indicate rounding/precision errors.
- **Metric-specific patterns:** Group errors by the financial metric being computed (ROI, WACC, NPV, etc.). The same metric may consistently use the wrong output convention.
- **Reasoning trace analysis:** Read the model's chain-of-thought for errors. Often the math is correct but the final step applies the wrong conversion (e.g., multiplying by 100 when it shouldn't, or not converting millions to base units).

**What makes an effective Formula prompt fix:**
- Explicit output scale convention rules: which metrics are expressed as decimals (raw ratio) vs percentages (×100). Include worked examples showing correct output for each.
- Unit conversion rules with examples (e.g., "3 billion = 3000000000.0, always express in base units").
- Precision rules (carry N significant figures through intermediate steps, round only at the end).
- Formula convention rules when multiple conventions exist (e.g., annuity-due vs ordinary annuity, discrete vs continuous compounding).
- Concrete "do X, not Y" examples grounded in actual errors from the data."""
