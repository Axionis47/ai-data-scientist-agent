"""
Playbook planner - rules-first parsing to map questions to fixed playbooks.

Playbooks:
A) OVERVIEW_PLAYBOOK - dataset summary and column overview
B) UNIVARIATE_PLAYBOOK - single column analysis
C) GROUPBY_PLAYBOOK - group by and aggregate
D) TREND_PLAYBOOK - time series trend analysis
E) CORRELATION_PLAYBOOK - correlation between numeric columns

If ambiguous or missing required info, returns NEEDS_CLARIFICATION with follow-up questions.
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PlaybookSelection:
    """Result of playbook selection."""
    playbook: str  # OVERVIEW, UNIVARIATE, GROUPBY, TREND, CORRELATION, NEEDS_CLARIFICATION
    confidence: float
    params: dict[str, Any]  # Parameters for the playbook
    clarification_questions: list[str]  # Non-empty if NEEDS_CLARIFICATION


# Pattern definitions for each playbook
OVERVIEW_PATTERNS = [
    r"\boverview\b",
    r"\bsummar(y|ize)\b",
    r"\bdescribe\s+(the\s+)?data",
    r"\bwhat\s+(columns|fields)\b",
    r"\bdata\s+dictionary\b",
    r"\bschema\b",
    r"\bwhat\s+is\s+(in\s+)?(this|the)\s+dataset",
]

UNIVARIATE_PATTERNS = [
    r"\bdistribution\s+of\b",
    r"\bstatistics\s+(for|of)\b",
    r"\bdescribe\s+column\b",
    r"\bsummary\s+of\s+(\w+)\b",
    r"\bhow\s+(many|much)\b.*\bcolumn\b",
    r"\bvalue\s+counts?\b",
    r"\btop\s+\d+\s+values?\b",
]

GROUPBY_PATTERNS = [
    r"\b(by|per|for\s+each)\s+\w+\b",
    r"\bgroup\s*by\b",
    r"\baverage\s+.+\s+(by|per)\b",
    r"\bsum\s+.+\s+(by|per)\b",
    r"\bcount\s+.+\s+(by|per)\b",
    r"\bbreakdown\b",
    r"\bsegment(ed)?\s+by\b",
]

TREND_PATTERNS = [
    r"\btrend\b",
    r"\bover\s+time\b",
    r"\btime\s+series\b",
    r"\bdaily\b|\bweekly\b|\bmonthly\b",
    r"\b(how\s+has|how\s+did)\s+.+\s+change\b",
    r"\bgrowth\b",
    r"\bhistorical\b",
]

CORRELATION_PATTERNS = [
    r"\bcorrelat(e|ion)\b",
    r"\brelationship\s+between\b",
    r"\bassociat(e|ion)\b",
    r"\bhow\s+(does|do)\s+.+\s+(relate|affect|impact)\b",
]


def _extract_column_mentions(question: str, column_names: list[str]) -> list[str]:
    """Extract column names mentioned in the question."""
    question_lower = question.lower()
    mentioned = []
    for col in column_names:
        # Check for exact match or quoted match
        if col.lower() in question_lower or f'"{col}"' in question:
            mentioned.append(col)
    return mentioned


def _extract_aggregation(question: str) -> str | None:
    """Extract aggregation function from question."""
    agg_patterns = {
        "sum": r"\bsum\b|\btotal\b",
        "mean": r"\baverage\b|\bmean\b|\bavg\b",
        "count": r"\bcount\b|\bnumber\s+of\b|\bhow\s+many\b",
        "min": r"\bminimum\b|\bmin\b|\blowest\b",
        "max": r"\bmaximum\b|\bmax\b|\bhighest\b",
        "median": r"\bmedian\b",
    }
    for agg, pattern in agg_patterns.items():
        if re.search(pattern, question, re.IGNORECASE):
            return agg
    return None


def _extract_frequency(question: str) -> str | None:
    """Extract time frequency from question."""
    if re.search(r"\bdaily\b|\bday\b|\bper\s+day\b", question, re.IGNORECASE):
        return "D"
    if re.search(r"\bweekly\b|\bweek\b|\bper\s+week\b", question, re.IGNORECASE):
        return "W"
    if re.search(r"\bmonthly\b|\bmonth\b|\bper\s+month\b", question, re.IGNORECASE):
        return "M"
    return None


def select_playbook(
    question: str,
    column_names: list[str],
    inferred_types: dict[str, str],
) -> PlaybookSelection:
    """
    Select the appropriate playbook based on the question.

    Args:
        question: User's natural language question
        column_names: List of column names in the dataset
        inferred_types: Dict of column name -> inferred type

    Returns:
        PlaybookSelection with playbook name, params, and any clarification questions
    """
    question_lower = question.lower()

    # Extract any mentioned columns
    mentioned_columns = _extract_column_mentions(question, column_names)

    # Check patterns in priority order
    # 1. Correlation (requires multiple numeric columns)
    for pattern in CORRELATION_PATTERNS:
        if re.search(pattern, question_lower):
            numeric_cols = [c for c, t in inferred_types.items() if t in ("integer", "float")]
            if len(mentioned_columns) >= 2:
                return PlaybookSelection(
                    playbook="CORRELATION",
                    confidence=0.9,
                    params={"columns": mentioned_columns},
                    clarification_questions=[],
                )
            elif len(numeric_cols) >= 2:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.8,
                    params={},
                    clarification_questions=[
                        f"Which numeric columns would you like to correlate? Available: {numeric_cols[:5]}",
                    ],
                )
            else:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.7,
                    params={},
                    clarification_questions=[
                        "Correlation requires at least 2 numeric columns. This dataset may not have enough numeric columns.",
                    ],
                )

    # 2. Trend analysis (requires date column)
    for pattern in TREND_PATTERNS:
        if re.search(pattern, question_lower):
            date_cols = [c for c, t in inferred_types.items() if t in ("datetime", "datetime_string")]
            freq = _extract_frequency(question) or "D"
            agg = _extract_aggregation(question) or "sum"

            if not date_cols:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.7,
                    params={},
                    clarification_questions=[
                        "I couldn't identify a date column. Which column contains dates?",
                    ],
                )

            # Find target column
            numeric_cols = [c for c, t in inferred_types.items() if t in ("integer", "float")]
            target_col = None
            for col in mentioned_columns:
                if col in numeric_cols:
                    target_col = col
                    break

            if not target_col and len(numeric_cols) == 1:
                target_col = numeric_cols[0]

            if not target_col:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.8,
                    params={},
                    clarification_questions=[
                        f"Which numeric column would you like to trend? Available: {numeric_cols[:5]}",
                    ],
                )

            return PlaybookSelection(
                playbook="TREND",
                confidence=0.9,
                params={
                    "date_col": date_cols[0],
                    "target_col": target_col,
                    "agg": agg,
                    "freq": freq,
                },
                clarification_questions=[],
            )

    # 3. Groupby aggregation
    for pattern in GROUPBY_PATTERNS:
        if re.search(pattern, question_lower):
            agg = _extract_aggregation(question) or "sum"

            # Find group column and target column
            group_col = None
            target_col = None

            categorical_cols = [c for c, t in inferred_types.items() if t == "string"]
            numeric_cols = [c for c, t in inferred_types.items() if t in ("integer", "float")]

            for col in mentioned_columns:
                if col in categorical_cols and not group_col:
                    group_col = col
                elif col in numeric_cols and not target_col:
                    target_col = col

            if not group_col:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.8,
                    params={},
                    clarification_questions=[
                        f"Which column would you like to group by? Categorical columns: {categorical_cols[:5]}",
                    ],
                )

            if not target_col:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.8,
                    params={},
                    clarification_questions=[
                        f"Which numeric column would you like to aggregate? Available: {numeric_cols[:5]}",
                    ],
                )

            return PlaybookSelection(
                playbook="GROUPBY",
                confidence=0.9,
                params={
                    "group_col": group_col,
                    "target_col": target_col,
                    "agg": agg,
                },
                clarification_questions=[],
            )

    # 4. Univariate analysis
    for pattern in UNIVARIATE_PATTERNS:
        if re.search(pattern, question_lower):
            if len(mentioned_columns) == 1:
                return PlaybookSelection(
                    playbook="UNIVARIATE",
                    confidence=0.9,
                    params={"column": mentioned_columns[0]},
                    clarification_questions=[],
                )
            elif len(mentioned_columns) > 1:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.8,
                    params={},
                    clarification_questions=[
                        f"You mentioned multiple columns: {mentioned_columns}. Which one should I analyze?",
                    ],
                )
            else:
                return PlaybookSelection(
                    playbook="NEEDS_CLARIFICATION",
                    confidence=0.8,
                    params={},
                    clarification_questions=[
                        f"Which column would you like to analyze? Available: {column_names[:5]}",
                    ],
                )

    # 5. Overview (catch-all for general questions)
    for pattern in OVERVIEW_PATTERNS:
        if re.search(pattern, question_lower):
            return PlaybookSelection(
                playbook="OVERVIEW",
                confidence=0.9,
                params={},
                clarification_questions=[],
            )

    # Default: If no pattern matches, try to be helpful
    # If they mention a single column, do univariate
    if len(mentioned_columns) == 1:
        return PlaybookSelection(
            playbook="UNIVARIATE",
            confidence=0.7,
            params={"column": mentioned_columns[0]},
            clarification_questions=[],
        )

    # If no columns mentioned, do overview
    if not mentioned_columns:
        return PlaybookSelection(
            playbook="OVERVIEW",
            confidence=0.6,
            params={},
            clarification_questions=[],
        )

    # Ambiguous case - ask for clarification
    return PlaybookSelection(
        playbook="NEEDS_CLARIFICATION",
        confidence=0.5,
        params={},
        clarification_questions=[
            "I'm not sure what analysis you'd like. Could you specify:",
            "- 'overview' for dataset summary",
            "- 'distribution of <column>' for univariate analysis",
            "- 'X by Y' for group-by aggregation",
            "- 'trend over time' for time series",
        ],
    )

