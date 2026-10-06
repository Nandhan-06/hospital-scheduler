# Hospital Appointment Coordinator Agent

A logistics-only scheduling agent: finds a specialist, checks insurance,
books a slot, and schedules a related lab test. It never diagnoses,
recommends treatment, or gives medical advice.

## Folder structure

```
hospital_scheduler/
  agents/
    manager.py       <- HospitalCoordinatorManager
    workers.py        <- BaseWorker, DoctorWorker, SlotWorker, LabTestWorker
    critic.py          <- AppointmentCritic
  tools/
    hospital_tools.py  <- load_hospital_data, search_doctors, search_slots, search_lab_tests
  data/
    sample_doctors.json
    sample_appointment_slots.json
    sample_lab_tests.json
  main.py
  utils.py              <- LLMClient (Gemini)
  .env                   <- DEMO_MODE=offline (or live)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # already done for you; defaults to DEMO_MODE=offline
```

To use the free Gemini API instead of the offline rule-based fallback:

1. Get a free key at https://aistudio.google.com/app/apikey
2. In `.env`, set `DEMO_MODE=live` and `GEMINI_API_KEY=your_key_here`

## Run

```bash
python main.py --request "I need to see a cardiologist within the next 3 days. I'm covered under Star Health insurance. Prefer morning slots. I also need a blood test done before the appointment. Keep my out-of-pocket cost under Rs 2,000."
```

Or just `python main.py` to use the built-in default request.

## How DEMO_MODE affects behavior

| Component | offline | live |
|---|---|---|
| `parse_request` | regex + specialty alias map | Gemini extracts structured fields, falls back to regex on failure |
| `DoctorWorker` / `SlotWorker` / `LabTestWorker` | deterministic scoring + templated blurb | same deterministic scoring (the LLM never overrides it) + Gemini-written one-line blurb, falls back to templated blurb on failure |
| `AppointmentCritic` | pure rule-based checks | pure rule-based checks (never LLM-based -- the 4 hard constraints must always be enforced exactly) |

**Design note:** the LLM is only ever used for parsing free text and for
narrating a decision the deterministic scorer already made. It is never
allowed to choose a doctor, slot, or lab test, and it never overrides a
critic check. This keeps hard constraints (insurance match, budget,
urgency) reliable regardless of what the API returns.

## Verified behavior

Running the default request produces:
- `requirements.insurance_provider = "Star Health"`, `urgency_days = 3`, `budget_cap_inr = 2000`
- Doctor picked: **D01** (Apollo Hospital), not D02 -- despite D02's higher
  4.8 rating, D02 does not accept Star Health, and the insurance penalty
  (±10) is built to always outweigh the rating spread (rating * 2, max 10).
- `cost.estimated_total_inr = 1850` (₹1,200 consultation + ₹650 lab test)
- `quality_score_after_reflexion = 90` (all 4 critic checks pass; formula: 95 - 5 generic note)

Try the failing-insurance demo:

```bash
python main.py --request "I need to see a cardiologist within the next 3 days. I'm covered under National Assurance insurance. Prefer morning slots. I also need a blood test done before the appointment. Keep my out-of-pocket cost under Rs 2,000."
```

Since neither sample doctor accepts "National Assurance", the critic flags
a high-severity insurance mismatch and `refine_appointment()` correctly
finds no in-network alternative to switch to. Note: with both doctors
equally penalized on insurance, the tie is broken by rating, so D02 (the
pricier doctor) gets picked -- which *also* trips the budget check. That's
a legitimate second failure, not a bug: it's a good moment to discuss what
the system should do when there's genuinely no valid option.

## Two parsing bugs this build avoids

1. **Specialty substring matching** -- `"cardiology" in text` fails on
   "I need to see a cardiologist" (different suffix). Fixed with an alias
   map (`SPECIALTY_ALIASES` in `agents/manager.py`).
2. **The "within X days" regex** -- `r"within\s+(\d+)\s*day"` fails on
   "within the next 3 days" because of the words in between. Fixed with
   `r"within\s+(?:the\s+next\s+)?(\d+)\s*day"`.

## Testing each stage in isolation

```bash
# Stage 1 -- tools only, no agents
python -c "
from tools.hospital_tools import load_hospital_data, search_doctors
data = load_hospital_data()
requirements = {'specialty_needed': 'Cardiology', 'insurance_provider': 'Star Health'}
for doc in search_doctors(requirements, data['doctors']):
    print(doc['doctor_id'], doc['selection_score'])
"

# Stage 2 -- one worker
python -c "
from utils import LLMClient
from tools.hospital_tools import load_hospital_data
from agents.workers import DoctorWorker
llm = LLMClient(demo_mode='offline')
data = load_hospital_data()
worker = DoctorWorker(llm, data['doctors'])
report = worker.run({'specialty_needed': 'Cardiology', 'insurance_provider': 'Star Health'})
print(report['recommendation']['doctor_id'])
"

# Stage 3 -- critic only
python -c "
from agents.critic import AppointmentCritic
from utils import LLMClient
critic = AppointmentCritic(LLMClient(demo_mode='offline'))
fake_requirements = {'insurance_provider': 'Random Co'}
fake_draft = {'selected_doctor': {'insurance_accepted': ['Star Health']}}
print(critic._check_insurance_coverage(fake_requirements, fake_draft))
"

# Stage 4/5 -- full pipeline
python main.py
```
