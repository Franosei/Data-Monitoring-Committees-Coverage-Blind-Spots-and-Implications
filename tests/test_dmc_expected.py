import json
import unittest

from codes.dmc_expected import DMCExpectedClassifier


class FakeDMCExpectedLLMClient:
    def __init__(self, serious_condition=False, serious_endpoint=False, confidence=0.9, rationale="LLM inference"):
        self.serious_condition = serious_condition
        self.serious_endpoint = serious_endpoint
        self.confidence = confidence
        self.rationale = rationale
        self.calls = 0
        self.last_payload = None

    def classify(self, messages):
        self.calls += 1
        self.last_payload = json.loads(messages[-1]["content"])
        return {
            "serious_or_life_threatening_condition": self.serious_condition,
            "serious_endpoint": self.serious_endpoint,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


class DMCExpectedClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = DMCExpectedClassifier()
        self.base_row = {
            "study_type": "Interventional",
            "phase": "Phase 2",
            "conditions": ["Mild Acne"],
            "keywords": [],
            "therapeutic_area": "Dermatology",
            "brief_title": "Low-risk behavioral trial",
            "official_title": "",
            "enrollment": 120,
            "minimum_age": "18 Years",
            "maximum_age": "64 Years",
            "intervention_types": ["BEHAVIORAL"],
            "primary_outcomes": ["Symptom score"],
            "secondary_outcomes": [],
            "primary_outcome_time_frames": ["12 weeks"],
            "secondary_outcome_time_frames": [],
            "start_date": "2020-01-01",
            "completion_date": "2020-12-31",
            "primary_completion_date": "2020-10-31",
        }

    def classify(self, **overrides):
        row = dict(self.base_row)
        row.update(overrides)
        return self.classifier.classify(row)

    def test_low_risk_interventional_trial_is_not_expected(self):
        result = self.classify()

        self.assertFalse(result.dmc_expected)
        self.assertEqual(result.dmc_expected_reason_count, 0)

    def test_phase3_alone_is_not_dmc_expected(self):
        # Phase 3 alone is a risk signal, not automatic evidence (Tier 2 requires a second feature).
        result = self.classify(phase="Phase 3")

        self.assertFalse(result.dmc_expected)
        self.assertTrue(result.dmc_expected_late_phase)
        # Broad single-feature rule still flags it (sensitivity analysis).
        self.assertTrue(result.dmc_expected_broad)

    def test_phase3_with_regulated_intervention_is_dmc_expected(self):
        # Late phase + regulated intervention satisfies Tier 2.
        result = self.classify(phase="Phase 3", intervention_types=["DRUG"])

        self.assertTrue(result.dmc_expected)
        self.assertEqual(result.dmc_expected_tier, "tier2")
        self.assertIn("late_phase", result.dmc_expected_reasons)
        self.assertIn("regulated_intervention", result.dmc_expected_reasons)

    def test_serious_condition_and_endpoint_are_expected_reasons(self):
        result = self.classify(
            conditions=["Metastatic lung cancer"],
            primary_outcomes=["Overall survival"],
        )

        self.assertTrue(result.dmc_expected)
        self.assertIn("serious_or_life_threatening_condition", result.dmc_expected_reasons)
        self.assertIn("serious_endpoint", result.dmc_expected_reasons)

    def test_vulnerable_age_large_enrollment_and_long_duration_are_captured(self):
        result = self.classify(
            enrollment=750,
            minimum_age="6 Months",
            maximum_age="80 Years",
            start_date="2020-01",
            completion_date="2023-01",
        )

        self.assertTrue(result.dmc_expected)
        self.assertIn("vulnerable_age_population", result.dmc_expected_reasons)
        self.assertIn("large_enrollment", result.dmc_expected_reasons)
        self.assertIn("long_duration", result.dmc_expected_reasons)
        self.assertGreaterEqual(result.trial_duration_months, 35.0)

    def test_regulated_intervention_alone_is_not_dmc_expected(self):
        # A drug trial alone is not automatic evidence — needs scale (Tier 3) or another feature.
        result = self.classify(intervention_types=["DRUG"])

        self.assertFalse(result.dmc_expected)
        self.assertTrue(result.dmc_expected_regulated_intervention)
        # Broad rule still flags it.
        self.assertTrue(result.dmc_expected_broad)

    def test_regulated_intervention_with_large_enrollment_is_dmc_expected(self):
        # Regulated intervention + large enrollment satisfies Tier 3.
        result = self.classify(intervention_types=["DRUG"], enrollment=600)

        self.assertTrue(result.dmc_expected)
        self.assertEqual(result.dmc_expected_tier, "tier3")
        self.assertIn("regulated_intervention", result.dmc_expected_reasons)
        self.assertIn("large_enrollment", result.dmc_expected_reasons)

    def test_maximum_age_elderly_does_not_fire_vulnerable_flag(self):
        # A trial open to adults up to age 70 is not exclusively elderly.
        result = self.classify(minimum_age="18 Years", maximum_age="70 Years")
        self.assertFalse(result.dmc_expected_vulnerable_population)

    def test_minimum_age_elderly_fires_vulnerable_flag(self):
        # A trial exclusively for patients aged 65+ is correctly flagged.
        result = self.classify(minimum_age="65 Years", maximum_age="85 Years")
        self.assertTrue(result.dmc_expected_vulnerable_population)

    def test_llm_fills_missing_serious_condition_or_endpoint(self):
        fake_llm = FakeDMCExpectedLLMClient(serious_condition=True, serious_endpoint=False)
        classifier = DMCExpectedClassifier(llm_client=fake_llm, llm_confidence_threshold=0.5)
        result = self.classify(
            study_type="Interventional",
            phase="Phase 2",
            conditions=["Rare autoimmune neurological disorder causing progressive paralysis"],
            primary_outcomes=["Time to functional decline and loss of independence"],
            keywords=[],
            intervention_types=["DRUG"],
        )

        self.assertTrue(result.dmc_expected)
        self.assertTrue(result.dmc_expected_serious_condition)
        self.assertFalse(result.dmc_expected_serious_endpoint)
        self.assertTrue(result.dmc_expected_llm_used)
        self.assertEqual(result.dmc_expected_llm_rationale, fake_llm.rationale)
        self.assertEqual(fake_llm.calls, 1)

    def test_non_interventional_study_keeps_flags_but_is_not_dmc_expected(self):
        result = self.classify(
            study_type="Observational",
            phase="Phase 3",
            conditions=["Stroke"],
            intervention_types=["DRUG"],
        )

        self.assertFalse(result.dmc_expected)
        self.assertTrue(result.dmc_expected_late_phase)
        self.assertTrue(result.dmc_expected_serious_condition)


if __name__ == "__main__":
    unittest.main()

