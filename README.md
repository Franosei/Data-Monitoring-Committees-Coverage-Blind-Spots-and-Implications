# Data Monitoring Committees: Coverage, Blind Spots, and Implications
# DMC Coverage Map: Independent Oversight in Clinical Trials

## Overview
This project investigates the use of **Data Monitoring Committees (DMCs)** in clinical trials registered on [ClinicalTrials.gov](https://clinicaltrials.gov) between **2010 and 2025**.  

It highlights where independent oversight is present, where it is lacking, and what this means for **patient safety**, **trial integrity**, and the **efficient use of resources**.  

The same principle is extended to **artificial intelligence in healthcare**: just as medicines and interventions require independent monitoring, so too should AI systems before they are evaluated or deployed in patient care.

---

## Why this matters
- **Protecting patients:** Trials with a DMC are more likely to stop early for safety or futility. This is not failure, but success – identifying risks or ineffectiveness earlier so fewer people are exposed unnecessarily.  
- **Avoiding wasted years:** Futile trials that run on without oversight cost sponsors, clinicians, and patients valuable time and resources.  
- **Regulatory alignment:** Both the **European Medicines Agency (EMA)** and the **US Food and Drug Administration (FDA)** advise that DMCs are essential in high-risk settings such as large, late-phase, or vulnerable-population trials.  
- **AI in healthcare:** Algorithms that guide diagnosis or treatment deserve the same independent evaluation before use. This work makes the case for “Algorithm Monitoring Committees” as an analogue to DMCs.

---

## Key Findings
- **Overall prevalence:** Only around **28%** of trials reported having a DMC.  
- **By phase:** Phase III and IV trials were more likely to use a DMC, but coverage was not universal. Early-phase studies rarely had DMCs, which is broadly consistent with regulatory guidance.  
- **By sponsor:** Academic sponsors were more likely to use DMCs than industry.  
- **By therapeutic area:** Oversight was patchy in high-risk fields such as neurology and psychiatry.  
- **Impact on outcomes:** Trials with a DMC were **twice as likely** to stop for safety reasons and **four times as likely** to stop for futility – clear evidence of earlier, better-informed decision-making.

---

## Visualisations
The repository includes high-resolution figures that are suitable for publication:
- **Heatmap:** DMC prevalence by sponsor and therapeutic area  
- **Trends by phase:** Annual DMC adoption (excluding N/A phases)  
- **Trends by sponsor:** Annual adoption by sponsor type  
- **Forest plot:** Odds of termination without a DMC, adjusted for phase and sponsor  

---

## Repository Structure
- `data/` → Cleaned datasets  
- `codes/` → Python scripts  
  - `metrics.py` → prevalence, gaps, outcomes, termination reasons  
  - `visualization.py` → plots and figures  
- `visualization/` → Generated images  

---

## Dependencies
- Python 3.9 or later  
- pandas  
- numpy  
- matplotlib  
- statsmodels  

