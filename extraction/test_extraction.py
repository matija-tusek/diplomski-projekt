"""Offline tests (no Ollama needed): python -m unittest extraction.test_extraction"""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from . import extractor
from .datasets import load_documents
from .evidence import locate
from .schemas import LlmCandidate, LlmJob

SOURCE = "LANGUAGES\nCroatian (native), English (C1), German (A2).\nDeveloped machine-learning models in Python."


class LocateTest(unittest.TestCase):
    def test_exact(self):
        ev = locate("English (C1)", SOURCE, "cv_text")
        self.assertEqual(SOURCE[ev["char_start"]:ev["char_end"]], "English (C1)")

    def test_relaxed_slip_returns_real_source_slice(self):
        ev = locate("German (A-2)", SOURCE, "cv_text")
        self.assertEqual(ev["text"], SOURCE[ev["char_start"]:ev["char_end"]])
        self.assertTrue(ev["text"].startswith("German (A2"))

    def test_case_and_hyphen(self):
        ev = locate("developed Machine Learning models", SOURCE, "cv_text")
        self.assertEqual(ev["text"], "Developed machine-learning models")

    def test_not_found(self):
        self.assertIsNone(locate("French (B1)", SOURCE, "cv_text"))
        self.assertIsNone(locate("  ", SOURCE, "cv_text"))


class MonthsTest(unittest.TestCase):
    today = date(2026, 11, 11)

    def test_month_range_inclusive(self):
        self.assertEqual(extractor._months("2025-03", "2025-08", None, self.today), 6)

    def test_present(self):
        self.assertEqual(extractor._months("2026-01", "present", None, self.today), 11)

    def test_year_only_is_unknown(self):
        self.assertIsNone(extractor._months("2024", "2025", None, self.today))

    def test_stated_duration_wins(self):
        self.assertEqual(extractor._months(None, None, 6, self.today), 6)


class LevelTest(unittest.TestCase):
    def ev(self, text):
        return {"text": text}

    def test_level_must_be_written(self):
        self.assertEqual(extractor._grounded_level("B2", self.ev("English (B2 or higher)")), "B2")
        self.assertEqual(extractor._grounded_level("A2", self.ev("German (A-2)")), "A2")
        self.assertIsNone(extractor._grounded_level("C1", self.ev("fluent English")))
        self.assertEqual(extractor._grounded_level("native", self.ev("Hrvatski (materinski)")), "native")
        self.assertIsNone(extractor._grounded_level("native", self.ev("Croatian, fluent")))

    def test_degree_must_be_named(self):
        self.assertEqual(extractor._grounded_degree("msc", self.ev("Current MSc student")), "msc")
        self.assertEqual(extractor._grounded_degree("msc", self.ev("Diplomski studij Računarstva")), "msc")
        self.assertIsNone(extractor._grounded_degree("msc", self.ev("Preddiplomski studij Računarstva")))
        self.assertIsNone(extractor._grounded_degree("msc", self.ev("Third-year undergraduate student")))
        self.assertIsNone(extractor._grounded_degree("bsc", self.ev("Status redovnog studenta je obavezan.")))


class DatasetsTest(unittest.TestCase):
    def test_csv_and_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "jobs.csv"
            csv_path.write_text('id,description\n7,"Python, SQL"\n8,\n', encoding="utf-8")
            self.assertEqual(list(load_documents(csv_path, "description", "id")), [("7", "Python, SQL")])
            jl = Path(d) / "cvs.jsonl"
            jl.write_text(json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n", encoding="utf-8")
            self.assertEqual(list(load_documents(jl, limit=1)), [("0", "a")])


def fake_llm(obj):
    return mock.patch.object(extractor, "call_llm", lambda prompt, text, schema: schema.model_validate(obj))


class PipelineTest(unittest.TestCase):
    """The LLM is mocked; checks grounding, dropping and contract validation."""

    def test_job(self):
        text = "ML Intern - Acme\nStrong Python skills. Docker is a plus. Hybrid in Zagreb. German is a plus."
        llm = {
            "title": "ML Intern", "occupation": "Machine Learning Engineer", "company": "Acme",
            "job_type": "internship", "location": {"city": "Zagreb", "country": "hr"},
            "remote_mode": {"evidence": "Hybrid in Zagreb", "value": "hybrid", "modality": "required"},
            "skills": [
                {"evidence": "Strong Python skills", "names": ["Python"], "modality": "required"},
                {"evidence": "Docker is a plus", "names": ["Docker"], "modality": "preferred"},
                {"evidence": "Kubernetes experience", "names": ["Kubernetes"], "modality": "required"},
            ],
            "education": {"evidence": None, "level": None, "field": None, "modality": "unspecified"},
            "student_status_required": {"evidence": None, "value": None, "modality": "unspecified"},
            # quote exists but states no number of years -> dropped
            "experience": {"evidence": "Docker is a plus", "min_years": 0, "modality": "preferred"},
            # level "B2" is not written in the evidence -> null
            "languages": [{"evidence": "German is a plus", "language": "German", "level": "B2",
                           "modality": "preferred"}],
        }
        LlmJob.model_validate(llm)
        with fake_llm(llm):
            profile, report = extractor.extract_job(text, "job_1", "test")
        self.assertEqual([s["name"] for s in profile["skills"]], ["Python", "Docker"])  # Kubernetes dropped
        self.assertEqual(report["dropped"][0]["item"], "skills: Kubernetes")
        self.assertIsNone(profile["experience"])
        self.assertIsNone(profile["languages"][0]["level"])
        self.assertIsNone(profile["education"])
        self.assertEqual(profile["location"]["country"], "HR")
        for s in profile["skills"]:
            ev = s["evidence"]
            self.assertEqual(text[ev["char_start"]:ev["char_end"]], ev["text"])

    def test_candidate_unknowns_are_null(self):
        text = "Ana\nTA (2024)\nRan lab sessions in C.\nSKILLS: Python, SQL\nEnglish (fluent)"
        llm = {
            "student_status": {"evidence": None, "is_student": None},
            "education": [],
            "skills": [{"evidence": "Python, SQL", "names": ["Python", "SQL"]},
                       {"evidence": "Python, SQL", "names": ["python"]}],
            "experience": [{"evidence": "TA (2024)", "title": "TA", "organization": None, "start": "2024",
                            "end": "2024", "stated_duration_months": None,
                            "description": "Ran lab sessions in C.", "skills_used": ["C", "Python"]}],
            "languages": [{"evidence": "English (fluent)", "language": "English", "level": None}],
            "preferences": {"target_roles": [], "interests": [], "locations": [], "remote_mode": None,
                            "job_types": []},
        }
        LlmCandidate.model_validate(llm)
        with fake_llm(llm):
            profile, _ = extractor.extract_candidate(text, "cand_1")
        self.assertEqual(profile["student_status"], {"is_student": None, "evidence": None})
        # duplicates removed; "C" comes from the experience description and is grounded there
        self.assertEqual([s["name"] for s in profile["skills"]], ["Python", "SQL", "C"])
        self.assertEqual(profile["skills"][2]["evidence"]["text"], "Ran lab sessions in C.")
        self.assertIsNone(profile["experience"][0]["months"])  # year-only dates
        self.assertIsNone(profile["languages"][0]["level"])
        self.assertIsNone(profile["preferences"]["target_roles"])


if __name__ == "__main__":
    unittest.main()
