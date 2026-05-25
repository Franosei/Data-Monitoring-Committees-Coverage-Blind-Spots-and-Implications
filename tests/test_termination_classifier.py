import json
import unittest

from codes.termination_classifier import (
    TERMINATION_CATEGORIES,
    OpenAITerminationLLMClient,
    TerminationReasonClassifier,
)


class FakeLLMClient:
    def __init__(self, category="administrative", confidence=0.81):
        self.category = category
        self.confidence = confidence
        self.calls = 0
        self.last_payload = None

    def classify(self, messages):
        self.calls += 1
        self.last_payload = json.loads(messages[-1]["content"])
        return {
            "termination_category": self.category,
            "confidence": self.confidence,
            "rationale": "Ambiguous text resolved by fallback.",
        }


class TerminationReasonClassifierTests(unittest.TestCase):
    def test_classifies_safety_reason_without_llm(self):
        classifier = TerminationReasonClassifier()

        result = classifier.classify(
            "Trial terminated due to serious adverse events and toxicity concerns"
        )

        self.assertEqual(result.termination_category, "safety")
        self.assertEqual(result.method, "rule")
        self.assertFalse(result.needs_llm)

    def test_keeps_futility_primary_when_recruitment_is_secondary(self):
        classifier = TerminationReasonClassifier()

        result = classifier.classify(
            "Recruitment challenges and results of interim futility analysis, "
            "which showed low likelihood to achieve the primary endpoint."
        )

        self.assertEqual(result.termination_category, "futility")
        self.assertIn("recruitment_failure", result.secondary_categories)
        self.assertTrue(result.needs_llm)

    def test_classifies_recruitment_failure(self):
        classifier = TerminationReasonClassifier()

        result = classifier.classify("Unable to recruit subjects")

        self.assertEqual(result.termination_category, "recruitment_failure")
        self.assertEqual(result.secondary_categories, [])

    def test_classifies_business_decision(self):
        classifier = TerminationReasonClassifier()

        result = classifier.classify("Company acquired and trial was not pursued")

        self.assertEqual(result.termination_category, "business_decision")

    def test_openai_response_text_extraction(self):
        client = OpenAITerminationLLMClient.__new__(OpenAITerminationLLMClient)
        response = type(
            "Resp",
            (),
            {
                "output_text": None,
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"termination_category":"efficacy","confidence":0.9,"rationale":"test"}',
                            }
                        ]
                    }
                ],
            },
        )()

        self.assertEqual(
            client._extract_response_text(response),
            '{"termination_category":"efficacy","confidence":0.9,"rationale":"test"}',
        )

    def test_llm_is_used_only_after_rule_fallback(self):
        fake_llm = FakeLLMClient()
        classifier = TerminationReasonClassifier(llm_client=fake_llm)

        result = classifier.classify(
            "Sponsor provided an ambiguous rationale.",
            overall_status="TERMINATED",
            trial_context={"brief_title": "Example Study"},
        )

        self.assertEqual(fake_llm.calls, 1)
        self.assertEqual(result.termination_category, "administrative")
        self.assertEqual(result.method, "llm")
        self.assertIn("valid_categories", fake_llm.last_payload)
        self.assertEqual(set(fake_llm.last_payload["valid_categories"]), set(TERMINATION_CATEGORIES))


if __name__ == "__main__":
    unittest.main()
