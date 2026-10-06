"""
main.py
-------
CLI entry point. Usage:

    python main.py --request "I need to see a cardiologist within the next 3 days..."
    python main.py                       # uses the default example request
    DEMO_MODE=live python main.py --request "..."   # calls Gemini for parsing + blurbs
"""

import argparse
import json

from utils import LLMClient
from agents.manager import HospitalCoordinatorManager

DEFAULT_REQUEST = (
    "I need to see a cardiologist within the next 3 days. I'm covered under Star Health "
    "insurance. Prefer morning slots. I also need a blood test done before the appointment. "
    "Keep my out-of-pocket cost under Rs 2,000."
)


def run_pipeline(request_text: str) -> dict:
    llm = LLMClient()  # reads DEMO_MODE / GEMINI_API_KEY / GEMINI_MODEL from env
    manager = HospitalCoordinatorManager(llm)
    return manager.run(request_text)


def main():
    parser = argparse.ArgumentParser(description="Hospital Appointment Coordinator Agent")
    parser.add_argument("--request", type=str, default=DEFAULT_REQUEST, help="Natural-language patient request")
    args = parser.parse_args()

    result = run_pipeline(args.request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
