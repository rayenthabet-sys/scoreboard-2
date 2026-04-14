"""
Pure scoring function — no external dependencies.
S = Σ(Ci × Wi)  where weights are Impact=0.4, Feasibility=0.3, Innovation=0.3
"""

WEIGHTS = {
    "impact": 0.4,
    "feasibility": 0.3,
    "innovation": 0.3,
}


def compute_score(
    impact_score: float,
    feasibility_score: float,
    innovation_score: float,
) -> float:
    """
    Compute final score out of 100.

    Args:
        impact_score:      0-100 sub-score for social impact (from Gemini).
        feasibility_score: 0-100 sub-score for technical feasibility (from Tavily).
        innovation_score:  0-100 sub-score for originality (from Llama / Gatekeeper).

    Returns:
        float: Weighted final score, rounded to 2 decimal places.
    """
    raw = (
        impact_score * WEIGHTS["impact"]
        + feasibility_score * WEIGHTS["feasibility"]
        + innovation_score * WEIGHTS["innovation"]
    )
    return round(min(max(raw, 0.0), 100.0), 2)
