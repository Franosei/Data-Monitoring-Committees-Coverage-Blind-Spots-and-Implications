# Data Monitoring Committees: Coverage, Blind Spots, and Implications

## Research Question

Among clinical trials registered on ClinicalTrials.gov from 2010 through 2025 with a final or inactive status, which trials have features matching FDA/EMA conditions where a Data Monitoring Committee (DMC) is recommended or ethically indicated, and how often do those trials report having a DMC?

This project deliberately uses the term **DMC expected by rule**, not "DMC required." The rule-based variable is an approximation of guidance-relevant risk features available in ClinicalTrials.gov registry data.

## Study Population

The analytic dataset includes ClinicalTrials.gov studies with:

- Start date from 2010-01-01 through 2025-12-31
- Overall status of `COMPLETED`, `TERMINATED`, `WITHDRAWN`, or `SUSPENDED`
- One row per unique NCT ID

The current dataset contains 281,540 unique trials.

## Extracted Variables

The downloader extracts the ClinicalTrials.gov fields needed to approximate oversight expectations:

- Trial phase and study type
- Intervention type and intervention names
- Sponsor type and lead sponsor
- Condition, keywords, and therapeutic area
- Enrollment size
- Minimum and maximum eligible age
- Start, primary completion, and completion dates
- Overall status
- Reported DMC presence
- `why_stopped` free text
- Primary and secondary outcome measures and time frames

## DMC Expected Rule

A trial is labeled `dmc_expected = True` when it is interventional and has at least one guidance-relevant feature:

- Phase 3 or Phase 4
- Serious or life-threatening condition, including oncology, major cardiovascular disease, sepsis, HIV/AIDS, stroke, organ failure, critical illness, and related high-risk diseases
- Child or elderly eligible population
- Large enrollment, currently defined as at least 500 participants
- Drug, biologic, device, genetic, radiation, or combination-product intervention
- Long duration, currently defined as at least 24 months from start to completion or primary completion
- Serious clinical endpoint, such as mortality, survival, serious adverse events, hospitalization, stroke, myocardial infarction, organ failure, or ventilation

Each row keeps the exact reasons that triggered the classification in `dmc_expected_reasons` and `dmc_expected_reason_labels`.

## Key Findings

- Overall DMC prevalence is 28.39%: 79,937 of 281,540 trials report a DMC.
- 197,359 trials are DMC expected by rule.
- Among DMC-expected trials, 131,707 do not report a DMC, a gap of 66.73%.
- Industry-sponsored DMC-expected trials have the largest reporting gap: 43,578 of 59,193 do not report a DMC.
- Trials with DMCs are more likely to be terminated, withdrawn, or suspended than trials without DMCs: 18.25% versus 12.35%.
- Structured `why_stopped` classification shows higher safety and futility stopping shares among DMC trials than non-DMC trials.

## Termination Reason Classification

Terminated, withdrawn, and suspended trials can include messy `why_stopped` text. The pipeline converts this text into predefined categories:

`safety`, `efficacy`, `futility`, `administrative`, `recruitment_failure`, `business_decision`, `other`, `unclear`.

Rules run first for reproducibility. Ambiguous, mixed, or uncategorized rows are written to:

`data/metrics/termination_reason_review_queue.csv`

Optional LLM fallback is off by default and can be enabled only for review-worthy cases:

```powershell
$env:ENABLE_LLM_TERMINATION_CLASSIFIER="true"
$env:OPENAI_MODEL="gpt-5-mini"
python codes/metrics.py
```

## Main Outputs

- `data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv` - cleaned ClinicalTrials.gov dataset
- `data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended_structured.csv` - row-level dataset with DMC-expected flags and termination categories
- `data/metrics/metric_dmc_expected_vs_actual.csv` - central expected-versus-reported DMC result
- `data/metrics/metric_dmc_expected_by_reason.csv` - which DMC-expected criteria drove classification
- `data/metrics/metric_dmc_expected_gap_by_phase.csv` - DMC gap among expected trials by phase
- `data/metrics/metric_dmc_expected_gap_by_sponsor.csv` - DMC gap among expected trials by sponsor
- `data/metrics/metric_dmc_expected_gap_by_therapeutic_area.csv` - DMC gap among expected trials by therapeutic area
- `data/metrics/metric_termination_category_by_dmc.csv` - structured stopping reasons by DMC status

## Repository Structure

- `codes/data_extraction.py` - ClinicalTrials.gov v2 downloader and field extraction
- `codes/dmc_expected.py` - rule-based DMC-expected classifier
- `codes/termination_classifier.py` - rule-first `why_stopped` classifier with optional LLM fallback
- `codes/metrics.py` - prevalence, DMC-expected gaps, outcomes, and termination metrics
- `codes/visualization.py` - publication-oriented figures
- `tests/` - unit tests for extraction, DMC expectation rules, and termination classification
- `visualization/` - generated figures

## Dependencies

- Python 3.9 or later
- pandas
- numpy
- matplotlib
- statsmodels
- requests
- openai, optional for LLM fallback only
