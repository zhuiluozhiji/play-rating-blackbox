from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Evidence:
    screenshot_path: Optional[str] = None
    html_path: Optional[str] = None
    log_path: Optional[str] = None


@dataclass
class SampleRecord:
    sample_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=utc_now_iso)
    strategy: str = "unknown"
    questionnaire_version: str = "unknown"
    category_path: List[str] = field(default_factory=list)
    answers_json: Dict[str, Any] = field(default_factory=dict)
    visible_questions: List[str] = field(default_factory=list)
    skipped_questions: List[str] = field(default_factory=list)
    submit_status: str = "not_submitted"
    result_age_rating: Optional[str] = None
    result_region_ratings: Dict[str, str] = field(default_factory=dict)
    content_descriptors: List[str] = field(default_factory=list)
    interactive_elements: List[str] = field(default_factory=list)
    certificate_or_result_id: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None
    evidence: Evidence = field(default_factory=Evidence)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleRecord":
        evidence = data.get("evidence") or {}
        if not isinstance(evidence, Evidence):
            data = dict(data)
            data["evidence"] = Evidence(**evidence)
        return cls(**data)


@dataclass(frozen=True)
class QuestionOption:
    value: str
    label: str
    risk_score: int = 0


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    question_type: str
    options: List[QuestionOption]
    theme: str = "general"
    parent_question_id: Optional[str] = None
    condition_to_show: Optional[Dict[str, Any]] = None
    ordered: bool = False


def default_question_schema() -> Dict[str, Any]:
    questions = [
        {
            "question_id": "violence",
            "text": "Does the app contain violent content?",
            "question_type": "single",
            "theme": "violence",
            "options": [
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "mild", "label": "Mild/cartoon violence", "risk_score": 1},
                {"value": "realistic", "label": "Realistic violence", "risk_score": 2},
                {"value": "graphic", "label": "Graphic violence", "risk_score": 4},
            ],
            "ordered": True,
        },
        {
            "question_id": "blood",
            "text": "Is blood or gore shown?",
            "question_type": "single",
            "theme": "violence",
            "parent_question_id": "violence",
            "condition_to_show": {"violence": ["realistic", "graphic"]},
            "options": [
                {"value": "not_visible", "label": "Not visible", "risk_score": 0},
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "blood", "label": "Blood", "risk_score": 1},
                {"value": "gore", "label": "Gore", "risk_score": 3},
            ],
            "ordered": True,
        },
        {
            "question_id": "sexual_content",
            "text": "Does the app contain sexual content or nudity?",
            "question_type": "single",
            "theme": "sexual_content",
            "options": [
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "suggestive", "label": "Suggestive themes", "risk_score": 1},
                {"value": "nudity", "label": "Nudity", "risk_score": 3},
                {"value": "explicit", "label": "Explicit sexual content", "risk_score": 5},
            ],
            "ordered": True,
        },
        {
            "question_id": "language",
            "text": "Does the app contain profanity or crude language?",
            "question_type": "single",
            "theme": "language",
            "options": [
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "mild", "label": "Mild language", "risk_score": 1},
                {"value": "strong", "label": "Strong language", "risk_score": 2},
            ],
            "ordered": True,
        },
        {
            "question_id": "drugs",
            "text": "Does the app reference drugs, alcohol, or tobacco?",
            "question_type": "single",
            "theme": "drugs",
            "options": [
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "reference", "label": "References", "risk_score": 1},
                {"value": "use", "label": "Depiction of use", "risk_score": 3},
            ],
            "ordered": True,
        },
        {
            "question_id": "gambling",
            "text": "Does the app contain gambling?",
            "question_type": "single",
            "theme": "gambling",
            "options": [
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "simulated", "label": "Simulated gambling", "risk_score": 2},
                {"value": "real_money", "label": "Real-money gambling", "risk_score": 5},
            ],
            "ordered": True,
        },
        {
            "question_id": "fear",
            "text": "Does the app contain frightening content?",
            "question_type": "single",
            "theme": "fear",
            "options": [
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "mild", "label": "Mild fear", "risk_score": 1},
                {"value": "intense", "label": "Intense fear", "risk_score": 2},
            ],
            "ordered": True,
        },
        {
            "question_id": "ugc",
            "text": "Does the app contain user-generated content or user interaction?",
            "question_type": "multi",
            "theme": "ugc_interaction",
            "options": [
                {"value": "none", "label": "None", "risk_score": 0},
                {"value": "user_generated_content", "label": "User-generated content", "risk_score": 1},
                {"value": "chat", "label": "Users can chat", "risk_score": 1},
                {"value": "location_sharing", "label": "Location sharing", "risk_score": 1},
            ],
        },
        {
            "question_id": "purchases",
            "text": "Does the app contain digital purchases?",
            "question_type": "binary",
            "theme": "interaction",
            "options": [
                {"value": "no", "label": "No", "risk_score": 0},
                {"value": "yes", "label": "Yes", "risk_score": 0},
            ],
        },
    ]
    return {"questionnaire_version": "synthetic_v1", "questions": questions}
