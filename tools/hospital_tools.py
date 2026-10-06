"""
tools/hospital_tools.py
------------------------
Pure, offline data-access functions. No LLM calls, no agent classes.
These simulate what would be live hospital/insurance API calls in
production. Every function is deterministic and testable in isolation.
"""

import json
import os
from datetime import date, datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_hospital_data(data_dir: str = DATA_DIR) -> dict:
    """Read the 3 JSON files into one dict: {doctors, slots, lab_tests}."""
    def _load(fname):
        with open(os.path.join(data_dir, fname), "r") as f:
            return json.load(f)

    return {
        "doctors": _load("sample_doctors.json"),
        "slots": _load("sample_appointment_slots.json"),
        "lab_tests": _load("sample_lab_tests.json"),
    }


def search_doctors(requirements: dict, doctors: list, top_k: int = 3) -> list:
    """Filter by specialty_needed, score by insurance match (dominant) + rating.

    Scoring: +10 if insurance_provider is in-network, -10 if not,
    plus rating * 2. The +-10 insurance swing is deliberately large enough
    to always outweigh the 0-5 point spread that rating*2 can produce
    (max rating 5.0 -> 10 points), matching the "hard constraint beats soft
    preference" pattern (avoid_red_eye in the travel planner).
    """
    specialty = requirements.get("specialty_needed")
    insurance = requirements.get("insurance_provider")

    candidates = [d for d in doctors if not specialty or d.get("specialty") == specialty]

    scored = []
    for d in candidates:
        insurance_accepted = d.get("insurance_accepted", [])
        in_network = insurance is not None and insurance in insurance_accepted
        insurance_score = 10 if in_network else (-10 if insurance is not None else 0)
        rating_score = d.get("rating", 0) * 2
        score = insurance_score + rating_score

        scored.append({
            **d,
            "in_network": in_network,
            "selection_score": round(score, 2),
        })

    scored.sort(key=lambda d: d["selection_score"], reverse=True)
    return scored[:top_k]


def _days_from_now(date_str: str, today: date = None) -> int:
    today = today or date.today()
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (target - today).days


def search_slots(requirements: dict, slots: list, doctor_id: str, top_k: int = 3) -> list:
    """Filter to one doctor's available slots, score by time-of-day match + recency."""
    preferred_time = requirements.get("preferred_time_of_day")

    candidates = [
        s for s in slots
        if s.get("doctor_id") == doctor_id and s.get("status") == "available"
    ]

    scored = []
    for s in candidates:
        days_out = _days_from_now(s["date"])
        time_match = 5 if (preferred_time and s.get("time_of_day") == preferred_time) else 0
        # Sooner slots score higher; recency_score shrinks as days_out grows.
        recency_score = max(0, 10 - days_out)
        score = time_match + recency_score

        scored.append({
            **s,
            "days_from_now": days_out,
            "selection_score": round(score, 2),
        })

    scored.sort(key=lambda s: s["selection_score"], reverse=True)
    return scored[:top_k]


def search_lab_tests(requirements: dict, lab_tests: list, top_k: int = 3) -> list:
    """Filter by linked_specialty, rank by price then turnaround (both ascending -> better)."""
    specialty = requirements.get("specialty_needed")

    candidates = [
        t for t in lab_tests
        if not specialty or t.get("linked_specialty") == specialty
    ]

    ranked = sorted(candidates, key=lambda t: (t.get("price_inr", 0), t.get("turnaround_hours", 0)))
    return ranked[:top_k]


if __name__ == "__main__":
    # Quick smoke test, per the build guide's Stage 1 instructions.
    data = load_hospital_data()
    requirements = {"specialty_needed": "Cardiology", "insurance_provider": "Star Health"}
    for doc in search_doctors(requirements, data["doctors"]):
        print(doc["doctor_id"], doc["selection_score"])
