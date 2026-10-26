"""Prompts for FiNER: financial named entity recognition (XBRL tagging)."""

# Initial prompt that every strategy starts from.
SEED_PROMPT = """\
You are an expert financial analyst specialized in Named Entity Recognition (NER) for financial documents.

Your task is to identify and classify financial entities in the given text according to the instruction provided.

Guidelines:
- Read the instruction carefully to understand what entity types are being requested
- Examine the input text methodically, identifying all relevant financial entities
- Classify each entity according to the specified categories
- If multiple entities are present, list them in order of appearance
- Use the exact category labels specified in the instruction
- Provide your classifications as a comma-separated list matching the expected format"""

# Task description shown to the SPO-RS and SPO-GA meta-model.
DESCRIPTION = """\
Financial Named Entity Recognition (XBRL Tagging)

Classify financial entities in SEC filing text by selecting the correct US GAAP XBRL taxonomy tag from a provided candidate list.

Input: A list of ~130 candidate XBRL tags + N sub-questions, each asking for the best tag for a specific entity in a sentence from an SEC filing.

Output: N comma-separated XBRL tag names, one per sub-question, in order.

Key challenge: Disambiguating within families of closely related tags using context clues in the sentence."""

# Richer task description shown to the SAGE agents.
AGENT_DESCRIPTION = """\
Financial Named Entity Recognition (XBRL Tagging)

Classify financial entities in SEC filing text by selecting the correct US GAAP XBRL taxonomy tag from a provided candidate list.

**Input format:** Each sample contains:
  - A list of ~130 candidate US GAAP XBRL tag names
  - N independent sub-questions (typically 2-6), each asking: 'What is the best tag for entity "<value>" in sentence: "<sentence>"?'
  - The sentence is an excerpt from an SEC 10-K or 10-Q filing

**Output format:** N comma-separated XBRL tag names, one per sub-question, in order. Example: 'LineOfCreditFacilityMaximumBorrowingCapacity,LettersOfCreditOutstandingAmount,DebtInstrumentInterestRateStatedPercentage'

**Key challenge:** Many XBRL tags belong to 'families' of closely related tags that differ in specificity or context (e.g., the LineOfCredit* family has 6+ tags for drawn amount, maximum capacity, remaining capacity, commitment fee, etc.). The primary difficulty is disambiguating within these families using context clues in the sentence (e.g., 'maximum borrowing capacity' vs 'outstanding borrowings' vs 'available under the facility'). Errors are rarely about identifying the wrong domain — they are almost always about picking the wrong tag within the correct tag family.

**Correctness:** Each sub-question answer must exactly match the expected tag string (case-sensitive). A sample is marked incorrect if ANY sub-question is wrong."""

# Task-specific error-analysis guidance for the SAGE agents.
ANALYSIS_GUIDANCE = """\
## Task-Specific Analysis Guidance (FiNER / XBRL Tagging)

This is an XBRL taxonomy disambiguation task. Almost all errors involve choosing the wrong tag within a family of related tags, not identifying the wrong domain entirely.

**High-value analysis patterns:**
- **Confusion matrix analysis:** Run `error_analysis.py --confusion` to get the top confused tag pairs. These pairs reveal which tag families the model struggles with most. A single tag family (e.g., LineOfCredit*) can account for 50-80+ errors.
- **Per-tag accuracy:** For each confused tag, compute how often the model gets it right vs wrong. A tag with 0% accuracy means the model NEVER predicts it correctly — the prompt likely lacks disambiguation rules for that tag family.
- **Sentence context patterns:** Read the model's reasoning traces for wrong predictions. Often the model identifies the right domain but lacks a decision rule for choosing between two similar tags (e.g., `PaymentsToAcquireBusinessesGross` vs `NetOfCashAcquired`).
- **Accounting standard versioning:** Some tag confusions arise from ASC 840 vs ASC 842 (old vs new lease accounting). Check if the model uses post-2018 tags for pre-2018 filings.

**What makes an effective FiNER prompt fix:**
- Definition blocks that list each tag in a family with its precise meaning and trigger phrases from the sentence (e.g., "when the sentence says 'outstanding borrowings' → use `LineOfCredit`; when it says 'maximum borrowing capacity' → use `LineOfCreditFacilityMaximumBorrowingCapacity`")
- Key disambiguation rules stated as concrete "if X then Y, not Z" with examples
- Focus on the highest-error tag families first — fixing one family of 6 tags can resolve 80+ errors"""
