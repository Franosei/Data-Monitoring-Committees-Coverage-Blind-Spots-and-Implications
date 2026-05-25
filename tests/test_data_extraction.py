import unittest

from codes.data_extraction import CTGV2DMCDownloader


class CTGV2DMCDownloaderExtractionTests(unittest.TestCase):
    def test_extract_record_includes_design_population_sponsor_intervention_and_outcomes(self):
        downloader = CTGV2DMCDownloader(storage_seen_path=None)
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                    "briefTitle": "Brief trial title",
                    "officialTitle": "Official trial title",
                },
                "oversightModule": {"oversightHasDmc": True},
                "statusModule": {
                    "overallStatus": "TERMINATED",
                    "whyStopped": "Stopped for futility after interim analysis",
                    "startDateStruct": {"date": "2020-01-01"},
                    "primaryCompletionDateStruct": {"date": "2022-01-01"},
                    "completionDateStruct": {"date": "2022-03-01"},
                },
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE3"],
                    "designInfo": {
                        "allocation": "RANDOMIZED",
                        "interventionModel": "PARALLEL",
                        "maskingInfo": {"masking": "DOUBLE"},
                        "primaryPurpose": "TREATMENT",
                    },
                    "enrollmentInfo": {"count": 450},
                },
                "eligibilityModule": {
                    "minimumAge": "18 Years",
                    "maximumAge": "75 Years",
                    "sex": "ALL",
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Example Pharma", "class": "INDUSTRY"},
                },
                "conditionsModule": {
                    "conditions": ["Lung Cancer"],
                    "keywords": ["mortality", "chemotherapy"],
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {"type": "DRUG", "name": "Examplemab"},
                        {"type": "DRUG", "name": "Examplemab"},
                        {"type": "PLACEBO", "name": "Placebo"},
                    ],
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "Overall survival", "timeFrame": "24 months"}
                    ],
                    "secondaryOutcomes": [
                        {"measure": "Serious adverse events", "timeFrame": "24 months"}
                    ],
                },
            }
        }

        record = downloader._extract_record(study)

        self.assertEqual(record["nct_id"], "NCT00000001")
        self.assertEqual(record["brief_title"], "Brief trial title")
        self.assertEqual(record["completion_date"], "2022-03-01")
        self.assertEqual(record["phase"], "Phase 3")
        self.assertEqual(record["allocation"], "RANDOMIZED")
        self.assertEqual(record["enrollment"], 450)
        self.assertEqual(record["sex"], "ALL")
        self.assertEqual(record["lead_sponsor_name"], "Example Pharma")
        self.assertEqual(record["lead_sponsor_class"], "INDUSTRY")
        self.assertEqual(record["intervention_types"], ["DRUG", "PLACEBO"])
        self.assertEqual(record["intervention_names"], ["Examplemab", "Placebo"])
        self.assertEqual(record["primary_outcomes"], ["Overall survival"])
        self.assertEqual(record["secondary_outcomes"], ["Serious adverse events"])
        self.assertEqual(record["therapeutic_area"], "Oncology")


if __name__ == "__main__":
    unittest.main()
