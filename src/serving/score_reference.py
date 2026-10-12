"""
Put a raw model score in context, in terms a business user understands.

The model's score is not a calibrated probability (it was trained with a
class weight), so "0.12" tells a person nothing. This module compares a
score with the scores of customers the model has already been evaluated on
(the held-out test set) and answers three plain questions:

    risk_percentile       "riskier than 22% of customers"
    risk_band             "Low", "Moderate", "High" or "Very high"
    observed_churn_rate   "in testing, 17% of customers in this band left"

The reference is built once per model by scripts/build_score_reference.py
and saved as model/score_reference.json. At serving time only the lookup
half of this module is used (it needs nothing but the standard library).
"""

import bisect
import json
import os

import numpy as np

REFERENCE_FILE = "score_reference.json"
BAND_NAMES = ("Low", "Moderate", "High", "Very high")

# A band with fewer customers than this gives a churn rate too noisy to show
MIN_BAND_SIZE = 100


def build_reference(
    scores, churned, threshold, low_max, very_high_min, run_id=None
) -> dict:
    """
    Describe how a model's scores relate to real outcomes.

    Args:
        scores: model scores for customers the model was NOT trained on
        churned: 1 if that customer actually churned, else 0
        threshold: the model's decision threshold (the High band starts here)
        low_max: scores below this are "Low"
        very_high_min: scores from this value up are "Very high"
        run_id: the run the model came from; checked when the reference is
            loaded next to a model, so a stale file cannot be used

    Returns:
        A JSON-serialisable dict (see ScoreReference for how it is used).

    Raises:
        ValueError: the band edges are out of order, a band holds too few
            customers, or the churn rate does not rise from band to band.
    """
    scores = np.asarray(scores, dtype=float)
    churned = np.asarray(churned, dtype=int)
    if not 0 < low_max < threshold < very_high_min < 1:
        raise ValueError(
            "Band edges must satisfy 0 < low_max < threshold < "
            f"very_high_min < 1 (got {low_max}, {threshold}, {very_high_min})"
        )

    edges = [0.0, low_max, threshold, very_high_min, 1.0]
    bands = []
    for i, name in enumerate(BAND_NAMES):
        lower, upper = edges[i], edges[i + 1]
        last = i == len(BAND_NAMES) - 1
        below_upper = (scores <= upper) if last else (scores < upper)
        in_band = (scores >= lower) & below_upper
        customers = int(in_band.sum())
        if customers < MIN_BAND_SIZE:
            raise ValueError(
                f"The '{name}' band holds only {customers} customers "
                f"(minimum {MIN_BAND_SIZE}); choose different band edges"
            )
        bands.append(
            {
                "name": name,
                "min": lower,
                "max": upper,
                "customers": customers,
                "churn_rate": round(float(churned[in_band].mean()), 4),
            }
        )

    rates = [band["churn_rate"] for band in bands]
    if rates != sorted(rates):
        raise ValueError(
            f"The churn rate does not rise with the band ({rates}); "
            "choose different band edges"
        )

    return {
        "run_id": run_id,
        "source": "held-out test set",
        "customers": int(len(scores)),
        "overall_churn_rate": round(float(churned.mean()), 4),
        "threshold": threshold,
        "bands": bands,
        "sorted_scores": [round(float(s), 6) for s in np.sort(scores)],
    }


def save_reference(reference: dict, path: str) -> None:
    """Write the reference; the long score list goes on a single line."""
    scores = reference["sorted_scores"]
    head = {k: v for k, v in reference.items() if k != "sorted_scores"}
    text = json.dumps(head, indent=2)[:-2]  # drop the closing "\n}"
    compact = json.dumps(scores, separators=(",", ":"))
    text += ',\n  "sorted_scores": ' + compact
    with open(path, "w") as f:
        f.write(text + "\n}\n")


class ScoreReference:
    """Looks a model score up against the saved reference."""

    def __init__(self, reference: dict):
        self.run_id = reference.get("run_id")
        self.overall_churn_rate = reference["overall_churn_rate"]
        self.bands = reference["bands"]
        self._scores = reference["sorted_scores"]
        self._band_starts = [band["min"] for band in self.bands]

    @classmethod
    def load(cls, model_dir: str):
        """Load model_dir/score_reference.json, or None if it is not there."""
        path = os.path.join(model_dir, REFERENCE_FILE)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return cls(json.load(f))

    def lookup(self, score: float) -> dict:
        """Context for one score."""
        # Share of reference customers with a lower score, as a whole percent
        below = bisect.bisect_left(self._scores, score)
        percentile = round(100 * below / len(self._scores))

        index = bisect.bisect_right(self._band_starts, score) - 1
        band = self.bands[max(index, 0)]

        return {
            "risk_percentile": percentile,
            "risk_band": band["name"],
            "observed_churn_rate": band["churn_rate"],
        }
