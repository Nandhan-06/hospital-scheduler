"""
agents/manager.py
------------------
HospitalCoordinatorManager: orchestrates parse -> workers -> draft -> critique
-> refine -> final output. This is the only file that talks to all the other
pieces; workers and critic have zero knowledge of each other.
"""

import re
import json

from tools.hospital_tools import load_hospital_data
from agents.workers import DoctorWorker, SlotWorker, LabTestWorker
from agents.critic import AppointmentCritic

SPECIALTY_ALIASES = {
    "Cardiology": ["cardiology", "cardiologist"],
    "Dermatology": ["dermatology", "dermatologist"],
    "Orthopedics": ["orthopedics", "orthopedist", "orthopaedic"],
    "Pediatrics": ["pediatrics", "pediatrician", "paediatrician"],
    "Neurology": ["neurology", "neurologist"],
    "ENT": ["ent", "ear nose throat", "otolaryngologist"],
    "General Medicine": ["general physician", "general medicine", "gp"],
}

# Bug 2 fix: tolerate filler words like "the next" between "within" and the digit.
URGENCY_PATTERN = re.compile(r"within\s+(?:the\s+next\s+)?(\d+)\s*day", re.IGNORECASE)
BUDGET_PATTERN = re.compile(r"(?:under|below|less than)\s*(?:rs\.?|inr|₹)\s*([\d,]+)", re.IGNORECASE)
INSURANCE_PATTERN = re.compile(
    r"(?:under|covered under|with)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+insurance", re.IGNORECASE
)


class HospitalCoordinatorManager:
    def __init__(self, llm_client, data=None):
        self.llm = llm_client
        self.data = data or load_hospital_data()

        self.doctor_worker = DoctorWorker(self.llm, self.data["doctors"])
        self.slot_worker = SlotWorker(self.llm, self.data["slots"])
        self.lab_worker = LabTestWorker(self.llm, self.data["lab_tests"])
        self.critic = AppointmentCritic(self.llm)

    # ---- step 1: parse -----------------------------------------------------

    def parse_request(self, request_text: str) -> dict:
        if self.llm.demo_mode == "live":
            try:
                return self._live_parse_request(request_text)
            except Exception:
                return self._offline_parse_request(request_text)
        return self._offline_parse_request(request_text)

    def _live_parse_request(self, request_text: str) -> dict:
        system = (
            "You extract structured scheduling requirements from a patient's "
            "natural-language appointment request. You never diagnose or give "
            "medical advice -- you only extract logistics fields."
        )
        user = (
            f'Request: "{request_text}"\n\n'
            "Return JSON with exactly these keys: specialty_needed (string, one of "
            f"{list(SPECIALTY_ALIASES.keys())}), urgency_days (int or null), "
            "insurance_provider (string or null), preferred_time_of_day "
            '("morning"|"afternoon"|"evening" or null), needs_lab_test (bool), '
            "budget_cap_inr (int or null), assumptions (list of strings)."
        )
        parsed = self.llm.complete_json(system, user)
        parsed.setdefault(
            "assumptions",
            ["Local sample data simulates hospital/insurance APIs.", "This is a scheduling demo, not medical advice."],
        )
        return parsed

    def _offline_parse_request(self, request_text: str) -> dict:
        text_lower = request_text.lower()

        # Bug 1 fix: alias map instead of literal substring match.
        specialty_needed = None
        for specialty, aliases in SPECIALTY_ALIASES.items():
            if any(alias in text_lower for alias in aliases):
                specialty_needed = specialty
                break

        urgency_match = URGENCY_PATTERN.search(text_lower)
        urgency_days = int(urgency_match.group(1)) if urgency_match else None

        budget_match = BUDGET_PATTERN.search(text_lower)
        budget_cap_inr = int(budget_match.group(1).replace(",", "")) if budget_match else None

        insurance_provider = None
        insurance_match = INSURANCE_PATTERN.search(request_text)
        if insurance_match:
            insurance_provider = insurance_match.group(1).strip()
        else:
            # Fallback: check known insurers from the dataset against the raw text.
            known_insurers = {ins for d in self.data["doctors"] for ins in d.get("insurance_accepted", [])}
            for ins in known_insurers:
                if ins.lower() in text_lower:
                    insurance_provider = ins
                    break

        preferred_time_of_day = None
        for tod in ("morning", "afternoon", "evening"):
            if tod in text_lower:
                preferred_time_of_day = tod
                break

        needs_lab_test = any(
            kw in text_lower for kw in ("blood test", "lab test", "lipid", "ecg", "test done")
        )

        return {
            "specialty_needed": specialty_needed,
            "urgency_days": urgency_days,
            "insurance_provider": insurance_provider,
            "preferred_time_of_day": preferred_time_of_day,
            "needs_lab_test": needs_lab_test,
            "budget_cap_inr": budget_cap_inr,
            "assumptions": [
                "Local sample data simulates hospital/insurance APIs.",
                "This is a scheduling demo, not medical advice.",
            ],
        }

    # ---- step 2: run workers -------------------------------------------------

    def run_workers(self, requirements: dict) -> dict:
        # Sequencing matters: slot search literally cannot run before a doctor
        # is chosen, since it needs that doctor's doctor_id.
        doctor_report = self.doctor_worker.run(requirements)

        doctor_id = None
        if doctor_report.get("recommendation"):
            doctor_id = doctor_report["recommendation"]["doctor_id"]

        if doctor_id:
            slot_report = self.slot_worker.run(requirements, doctor_id=doctor_id)
        else:
            slot_report = {"worker": "slot_worker", "status": "no_match", "recommendation": None, "alternatives": []}

        lab_report = self.lab_worker.run(requirements)

        return {"doctor": doctor_report, "slot": slot_report, "lab": lab_report}

    # ---- step 3: build draft --------------------------------------------------

    def create_draft_appointment(self, worker_reports: dict) -> dict:
        doctor = (worker_reports["doctor"].get("recommendation") or {})
        slot = (worker_reports["slot"].get("recommendation") or {})
        lab_test = worker_reports["lab"].get("recommendation")  # may legitimately be None

        consultation_fee = doctor.get("consultation_fee_inr", 0) or 0
        lab_price = (lab_test or {}).get("price_inr", 0) or 0
        estimated_total = consultation_fee + lab_price

        return {
            "status": "draft",
            "selected_doctor": doctor or None,
            "selected_slot": slot or None,
            "selected_lab_test": lab_test,
            "cost": {
                "consultation_fee_inr": consultation_fee,
                "lab_test_inr": lab_price,
                "estimated_total_inr": estimated_total,
            },
        }

    # ---- step 4: refine on critique -------------------------------------------

    def refine_appointment(self, requirements: dict, draft: dict, critique: dict, worker_reports: dict):
        """If the critic flagged an insurance mismatch, search the doctor
        worker's other options for an in-network alternative. Returns
        (refined_draft, changes_after_critique list).
        """
        changes = []
        issue_types = {issue["issue"] for issue in critique["issues"]}
        insurance_flagged = any("insurance" in issue.lower() for issue in issue_types)

        if not insurance_flagged:
            changes.append("Kept core choices and made assumptions/trade-offs explicit.")
            return draft, changes

        alternatives = worker_reports["doctor"].get("alternatives", [])
        insurance_provider = requirements.get("insurance_provider")
        alt_doctor = next(
            (d for d in alternatives if insurance_provider in d.get("insurance_accepted", [])),
            None,
        )

        if not alt_doctor:
            changes.append("Kept core choices and made assumptions/trade-offs explicit.")
            return draft, changes

        # Re-run slot search for the alternative doctor and rebuild the draft.
        new_slot_report = self.slot_worker.run(requirements, doctor_id=alt_doctor["doctor_id"])
        new_worker_reports = {
            "doctor": {"recommendation": alt_doctor},
            "slot": new_slot_report,
            "lab": worker_reports["lab"],
        }
        new_draft = self.create_draft_appointment(new_worker_reports)
        changes.append(
            f"Switched to {alt_doctor['doctor_id']} because the original pick was not in-network for {insurance_provider}."
        )
        return new_draft, changes

    # ---- step 5: run -----------------------------------------------------------

    def run(self, request_text: str) -> dict:
        requirements = self.parse_request(request_text)
        worker_reports = self.run_workers(requirements)
        draft = self.create_draft_appointment(worker_reports)
        critique = self.critic.critique(requirements, draft)
        refined_draft, changes = self.refine_appointment(requirements, draft, critique, worker_reports)
        final_critique = self.critic.critique(requirements, refined_draft) if refined_draft is not draft else critique

        final_appointment = {
            **refined_draft,
            "status": "final",
            "quality_score_after_reflexion": final_critique["quality_score"],
            "critic_summary": final_critique["summary"],
        }
        # keep field order close to the spec's example
        final_appointment = {
            "status": final_appointment["status"],
            "selected_doctor": final_appointment["selected_doctor"],
            "selected_slot": final_appointment["selected_slot"],
            "selected_lab_test": final_appointment["selected_lab_test"],
            "cost": final_appointment["cost"],
            "quality_score_after_reflexion": final_appointment["quality_score_after_reflexion"],
            "critic_summary": final_appointment["critic_summary"],
        }

        return {
            "requirements": requirements,
            "final_appointment": final_appointment,
            "changes_after_critique": changes,
            "critic_issues": final_critique["issues"],
            "final_notes": [
                "This is a scheduling demo using sample data, not medical advice.",
                "In production, each local tool can be replaced by a live hospital/insurance API adapter.",
            ],
        }
