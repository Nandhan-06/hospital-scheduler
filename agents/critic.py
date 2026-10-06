"""
agents/critic.py
-----------------
AppointmentCritic: reflexion-style critic that inspects a draft appointment
against the parsed requirements and flags hard-constraint violations.

Each _check_... method takes (requirements, draft) and returns either
None (no problem) or an issue dict:
    {"severity": "high"|"medium"|"low", "issue": "...", "fix": "..."}

Scoring: start at 95, subtract 20/10/5 per high/medium/low issue, then
always subtract a further 5 for the generic "nothing is ever perfect"
note (this matches the reference course pattern: an issue-free draft
lands at 90, one high-severity issue lands at 70). Floor of 60.
"""

SEVERITY_WEIGHTS = {"high": 20, "medium": 10, "low": 5}


class AppointmentCritic:
    def __init__(self, llm_client):
        self.llm = llm_client

    # ---- individual checks -------------------------------------------------

    def _check_insurance_coverage(self, requirements: dict, draft: dict):
        """Hard constraint: selected doctor must accept the stated insurer."""
        insurance_provider = requirements.get("insurance_provider")
        doctor = draft.get("selected_doctor") or {}
        if not insurance_provider:
            return None
        accepted = doctor.get("insurance_accepted", [])
        if insurance_provider not in accepted:
            return {
                "severity": "high",
                "issue": "Selected doctor does not accept the patient's insurance provider.",
                "fix": "Select a doctor in-network for the stated insurance provider.",
            }
        return None

    def _check_urgency_window(self, requirements: dict, draft: dict):
        """Hard constraint: the chosen slot must fall within urgency_days."""
        urgency_days = requirements.get("urgency_days")
        slot = draft.get("selected_slot") or {}
        if urgency_days is None:
            return None
        days_out = slot.get("days_from_now")
        if days_out is None or days_out > urgency_days:
            return {
                "severity": "high",
                "issue": (
                    f"Earliest available slot ({days_out} day(s) out) falls outside "
                    f"the {urgency_days}-day urgency window."
                ),
                "fix": "Look for an earlier slot, or with a different in-network doctor.",
            }
        return None

    def _check_lab_test_included(self, requirements: dict, draft: dict):
        """Coverage check: if a lab test was requested, one must be scheduled."""
        if not requirements.get("needs_lab_test", False):
            return None
        if not draft.get("selected_lab_test"):
            return {
                "severity": "medium",
                "issue": "A lab test was requested but none was scheduled.",
                "fix": "Schedule a lab test linked to the patient's specialty before the appointment.",
            }
        return None

    def _check_consultation_budget(self, requirements: dict, draft: dict):
        """Budget check: consultation fee + lab test cost must stay under budget_cap_inr."""
        budget_cap = requirements.get("budget_cap_inr")
        if budget_cap is None:
            return None
        cost = draft.get("cost", {})
        total = cost.get("estimated_total_inr")
        if total is None or total > budget_cap:
            return {
                "severity": "high",
                "issue": f"Estimated total cost (₹{total}) exceeds the patient's budget cap (₹{budget_cap}).",
                "fix": "Choose a lower-fee doctor or a cheaper lab test to fit the budget.",
            }
        return None

    # ---- aggregate critique --------------------------------------------------

    def critique(self, requirements: dict, draft: dict) -> dict:
        checks = [
            self._check_insurance_coverage,
            self._check_urgency_window,
            self._check_lab_test_included,
            self._check_consultation_budget,
        ]
        issues = [issue for issue in (check(requirements, draft) for check in checks) if issue]

        score = 95
        for issue in issues:
            score -= SEVERITY_WEIGHTS.get(issue["severity"], 5)
        score -= 5  # generic "nothing is ever perfect" note
        score = max(60, score)

        summary = "Ready with minor polish" if not issues else "Issues found -- refinement needed"

        return {
            "issues": issues,
            "quality_score": score,
            "summary": summary,
        }
