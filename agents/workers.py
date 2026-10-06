"""
agents/workers.py
------------------
BaseWorker + 3 domain workers: DoctorWorker, SlotWorker, LabTestWorker.

Each worker's job: run a tool search, then produce a small "report" dict
shaped like:
    {
        "worker": <name>,
        "status": "ok" | "not_requested" | "no_match",
        "recommendation": <top candidate dict or None>,
        "alternatives": [<other candidates>],
        "source": "offline" | "live",
        "blurb": <one-line natural-language note>,
    }

In offline mode the blurb is templated. In live mode the worker asks
Gemini to turn the same search results into a short natural-language
blurb -- the LLM never chooses the candidate, it only narrates the
choice the deterministic scorer already made. This keeps the hard
constraints (insurance, budget, urgency) enforced by code, not by an
LLM that could hallucinate around them.
"""

from tools.hospital_tools import search_doctors, search_slots, search_lab_tests


class BaseWorker:
    worker_name = "base_worker"
    live_system_prompt = "You are a helpful assistant that summarizes a search result."

    def __init__(self, llm_client, dataset):
        self.llm = llm_client
        self.dataset = dataset

    # Subclasses implement this to call the right tools.hospital_tools function.
    def search(self, requirements: dict, **kwargs) -> list:
        raise NotImplementedError

    # Subclasses implement this to build the offline (rule-based) report.
    def _offline_report(self, requirements: dict, candidates: list) -> dict:
        raise NotImplementedError

    def run(self, requirements: dict, **kwargs) -> dict:
        candidates = self.search(requirements, **kwargs)

        if self.llm.demo_mode != "live":
            return self._offline_report(requirements, candidates)

        try:
            return self._live_report(requirements, candidates)
        except Exception:
            # Any Gemini failure (quota, network, bad JSON) -> fall back to
            # the deterministic offline report rather than break the pipeline.
            report = self._offline_report(requirements, candidates)
            report["source"] = "offline_fallback"
            return report

    def _live_report(self, requirements: dict, candidates: list) -> dict:
        offline = self._offline_report(requirements, candidates)
        user_payload = (
            f"Requirements: {requirements}\n"
            f"Top candidate already selected by deterministic scoring: {offline.get('recommendation')}\n"
            f"Other candidates: {offline.get('alternatives')}\n"
            "Write a one-sentence, friendly blurb explaining why this candidate fits. "
            "Return JSON: {\"blurb\": \"...\"}"
        )
        result = self.llm.complete_json(self.live_system_prompt, user_payload)
        offline["blurb"] = result.get("blurb", offline.get("blurb", ""))
        offline["source"] = "live"
        return offline


class DoctorWorker(BaseWorker):
    worker_name = "doctor_worker"
    live_system_prompt = (
        "You are the doctor-matching assistant for a hospital scheduling system. "
        "You never diagnose or give medical advice -- you only explain, in one "
        "sentence, why a given doctor was matched to a patient's stated specialty "
        "and insurance requirements."
    )

    def search(self, requirements: dict, **kwargs) -> list:
        return search_doctors(requirements, self.dataset, top_k=3)

    def _offline_report(self, requirements: dict, candidates: list) -> dict:
        if not candidates:
            return {
                "worker": self.worker_name,
                "status": "no_match",
                "recommendation": None,
                "alternatives": [],
                "source": "offline",
                "blurb": f"No {requirements.get('specialty_needed', 'matching')} doctors found in the dataset.",
            }

        top, rest = candidates[0], candidates[1:]
        in_network_note = "in-network" if top.get("in_network") else "NOT in-network"
        return {
            "worker": self.worker_name,
            "status": "ok",
            "recommendation": top,
            "alternatives": rest,
            "source": "offline",
            "blurb": (
                f"Top match: {top['doctor_id']} at {top.get('hospital')} "
                f"(rating {top.get('rating')}, {in_network_note} for "
                f"{requirements.get('insurance_provider', 'the stated insurer')})."
            ),
        }


class SlotWorker(BaseWorker):
    """Needs a doctor_id before it can search -- that's only known after
    DoctorWorker has already produced a recommendation, so run() takes an
    extra required argument instead of matching BaseWorker's single-arg
    signature. This is a deliberate, documented exception to the pattern.
    """

    worker_name = "slot_worker"
    live_system_prompt = (
        "You are the appointment-slot assistant for a hospital scheduling system. "
        "You only explain, in one sentence, why a given slot was chosen based on "
        "time-of-day preference and how soon it falls."
    )

    def search(self, requirements: dict, doctor_id: str = None, **kwargs) -> list:
        if doctor_id is None:
            raise ValueError("SlotWorker.search requires doctor_id")
        return search_slots(requirements, self.dataset, doctor_id, top_k=3)

    def run(self, requirements: dict, doctor_id: str = None) -> dict:
        if doctor_id is None:
            raise ValueError("SlotWorker.run requires doctor_id (chosen by DoctorWorker first)")
        candidates = self.search(requirements, doctor_id=doctor_id)

        if self.llm.demo_mode != "live":
            return self._offline_report(requirements, candidates)
        try:
            return self._live_report(requirements, candidates)
        except Exception:
            report = self._offline_report(requirements, candidates)
            report["source"] = "offline_fallback"
            return report

    def _offline_report(self, requirements: dict, candidates: list) -> dict:
        urgency_days = requirements.get("urgency_days")
        if not candidates:
            return {
                "worker": self.worker_name,
                "status": "no_match",
                "recommendation": None,
                "alternatives": [],
                "source": "offline",
                "blurb": "No available slots found for this doctor.",
            }

        top, rest = candidates[0], candidates[1:]
        within_window = urgency_days is None or top["days_from_now"] <= urgency_days
        window_note = "within" if within_window else "OUTSIDE"
        return {
            "worker": self.worker_name,
            "status": "ok",
            "recommendation": top,
            "alternatives": rest,
            "source": "offline",
            "blurb": (
                f"Earliest matching slot: {top['date']} {top['time']} "
                f"({top['days_from_now']} day(s) out, {window_note} the "
                f"{urgency_days}-day urgency window)."
            ),
        }


class LabTestWorker(BaseWorker):
    worker_name = "lab_test_worker"
    live_system_prompt = (
        "You are the lab-test assistant for a hospital scheduling system. "
        "You only explain, in one sentence, why a given lab test was recommended "
        "based on price and turnaround time. You never give medical advice."
    )

    def search(self, requirements: dict, **kwargs) -> list:
        return search_lab_tests(requirements, self.dataset, top_k=3)

    def run(self, requirements: dict, **kwargs) -> dict:
        # New pattern vs. the other two workers: conditionally relevant.
        if not requirements.get("needs_lab_test", False):
            return {
                "worker": self.worker_name,
                "status": "not_requested",
                "recommendation": None,
                "alternatives": [],
                "source": "offline",
                "blurb": "No lab test requested -- skipped.",
            }
        return super().run(requirements, **kwargs)

    def _offline_report(self, requirements: dict, candidates: list) -> dict:
        if not candidates:
            return {
                "worker": self.worker_name,
                "status": "no_match",
                "recommendation": None,
                "alternatives": [],
                "source": "offline",
                "blurb": f"No lab tests found for {requirements.get('specialty_needed', 'the requested specialty')}.",
            }

        top, rest = candidates[0], candidates[1:]
        return {
            "worker": self.worker_name,
            "status": "ok",
            "recommendation": top,
            "alternatives": rest,
            "source": "offline",
            "blurb": (
                f"Recommended pre-consultation test: {top['test_name']} "
                f"(₹{top['price_inr']}, {top['turnaround_hours']}h turnaround)."
            ),
        }
