"""
Rule-first classification of ClinicalTrials.gov whyStopped text.

The goal is to keep the analysis reproducible by default. Rules classify clear,
high-signal reasons locally; an optional LLM client can be attached only for
ambiguous or mixed cases.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


TERMINATION_CATEGORIES: Tuple[str, ...] = (
    "safety",
    "efficacy",
    "futility",
    "administrative",
    "recruitment_failure",
    "business_decision",
    "mixed_reasons",  # 3+ distinct categories co-occur; LLM picks the dominant one
    "uncertain",      # text exists but no rule matched; LLM reclassifies
    "unclear",        # text is missing, circular, or too generic
)


@dataclass(frozen=True)
class RulePattern:
    """A regex and a readable term label for audit output."""

    regex: str
    label: str


@dataclass(frozen=True)
class CategoryRule:
    """A set of patterns that imply one termination category."""

    category: str
    patterns: Tuple[RulePattern, ...]


@dataclass
class TerminationClassification:
    """Structured result for one whyStopped string."""

    termination_category: str
    confidence: float
    method: str
    matched_terms: List[str]
    secondary_categories: List[str]
    needs_llm: bool
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for pandas/DataFrame use."""
        data = asdict(self)
        data["matched_terms"] = "; ".join(self.matched_terms)
        data["secondary_categories"] = "; ".join(self.secondary_categories)
        return data


class TerminationLLMClient(Protocol):
    """Small protocol so any LLM provider can be plugged in."""

    def classify(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        """Return a JSON-like mapping with termination_category and confidence."""


class OpenAITerminationLLMClient:
    """
    Optional OpenAI adapter.

    This is deliberately imported lazily so the deterministic rule classifier
    does not require the openai package or a network call.
    """

    RESPONSE_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "termination_category": {"type": "string", "enum": list(TERMINATION_CATEGORIES)},
            "confidence": {"type": "number", "description": "Confidence from 0 to 1."},
            "rationale": {"type": "string", "description": "Brief rationale under 20 words."},
        },
        "required": ["termination_category", "confidence", "rationale"],
    }

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # type: ignore

        self.model = model
        self.client = OpenAI()

    def classify(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.model,
                input=list(messages),
                temperature=0,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "termination_reason_classification",
                        "schema": self.RESPONSE_SCHEMA,
                        "strict": True,
                    }
                },
            )
            content = self._extract_response_text(response) or "{}"
            return json.loads(content)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "termination_reason_classification",
                    "schema": self.RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
            max_tokens=220,
        )
        content = self._extract_chat_response_text(response) or "{}"
        return json.loads(content)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "output_text") and response.output_text:
            return str(response.output_text)
        output = getattr(response, "output", None)
        if isinstance(output, str):
            return output
        if isinstance(output, (list, tuple)):
            parts: List[str] = []
            for item in output:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, (list, tuple)):
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict):
                            parts.append(str(block.get("text", "")))
                        else:
                            parts.append(str(block))
                elif content is not None:
                    parts.append(str(content))
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _extract_chat_response_text(response: Any) -> str:
        if response is None:
            return ""
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")
        if not message:
            return ""
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return str(content or "").strip()


class TerminationReasonClassifier:
    """
    Classify whyStopped text into a pre-specified category schema.

    Rules are used first. If an LLM client is supplied, it is called only when
    the rule result is low-confidence, ambiguous, or uncategorized.
    """

    CATEGORY_PRIORITY: Tuple[str, ...] = (
        "safety",
        "efficacy",
        "futility",
        "recruitment_failure",
        "business_decision",
        "administrative",
    )

    SCIENTIFIC_CATEGORIES = {"safety", "efficacy", "futility"}
    FALLBACK_LLM_CATEGORIES = {"mixed_reasons", "uncertain", "unclear"}

    RULES: Tuple[CategoryRule, ...] = (
        CategoryRule(
            "safety",
            (
                RulePattern(r"\bsafety\b", "safety"),
                RulePattern(r"\bsafety concerns?\b", "safety concern"),
                RulePattern(r"\badverse (event|events|reaction|reactions|effect|effects)\b", "adverse event"),
                RulePattern(r"\bserious adverse\b", "serious adverse event"),
                RulePattern(r"\bsae(s)?\b", "SAE"),
                RulePattern(r"\btoxicit(y|ies)\b", "toxicity"),
                RulePattern(r"\bdeath(s)?\b", "death"),
                RulePattern(r"\bmortality\b", "mortality"),
                RulePattern(r"\bharm(s|ful)?\b", "harm"),
                RulePattern(r"\bunacceptable risk\b", "unacceptable risk"),
                RulePattern(r"\brisk[- ]benefit\b", "risk-benefit"),
                # DSMB/DMC stops (most DSMB stops are safety-driven)
                # Require actual stopping language (not just "recommendation") to avoid tagging ambiguous DSMB mentions
                RulePattern(r"\b(dsmb|dmb|dsmc)\b.{0,80}\b(halt\w*|stop\w*|terminat\w*|suspend\w*)\b", "DSMB stopped"),
                RulePattern(r"\b(halt\w*|stop\w*|terminat\w*|suspend\w*)\b.{0,80}\b(dsmb|dmb|dsmc)\b", "halted by DSMB"),
                RulePattern(r"\bhalting criteria\b", "halting criteria"),
                # Reactogenicity / unexpected clinical events
                RulePattern(r"\breactogeni\w*\b", "reactogenicity"),
                RulePattern(r"\bunexpected.{0,60}\b(serious|severe|adverse)\b", "unexpected serious event"),
                RulePattern(r"\bhigher rate of.{0,60}\b(rejection|adverse|toxicit)\b", "higher rate adverse"),
                RulePattern(r"\b(acute |)rejection.{0,40}\b(high|higher|elevated|increased)\b", "elevated rejection"),
                RulePattern(r"\bdose[- ]limiting toxicit\w*\b", "dose-limiting toxicity"),
                RulePattern(r"\bstopping rule\b", "stopping rule"),
                RulePattern(r"\b(gi|gastrointestinal).{0,20}tolerabilit\w*\b", "GI tolerability"),
                RulePattern(r"\brisk mitigation\b", "risk mitigation"),
                RulePattern(r"\bvolume depletion\b", "volume depletion"),
                RulePattern(r"\belectrolyte (abnormal\w*|imbalanc\w*)\b", "electrolyte abnormality"),
                RulePattern(r"\bconcern.{0,60}\b(safety|adverse|depletion|abnormal|toxicit|hypersensitiv)\b", "safety concern (general)"),
                RulePattern(r"\bhypersensitiv\w*\b", "hypersensitivity"),
                RulePattern(r"\bvascular.{0,30}\bocclusion\b", "vascular occlusion"),
                RulePattern(r"\bocclusion.{0,30}\b(patient|subject|participant)\b", "occlusion in patient"),
                RulePattern(r"\bdiscontinuation rate\b", "discontinuation rate"),
                RulePattern(r"\bhypoxemia\b", "hypoxemia"),
                RulePattern(r"\bexcess\b.{0,60}\b(adverse|health service|hospitali\w*|utilization)\b", "excess adverse utilization"),
                RulePattern(r"\bcomplication\w*.{0,20}\b(to|with|in|of).{0,20}\b(procedure|intervention|treatment|surgery)\b", "procedure complication"),
            ),
        ),
        CategoryRule(
            "efficacy",
            (
                RulePattern(r"\boverwhelming efficacy\b", "overwhelming efficacy"),
                RulePattern(r"\bearly efficacy\b", "early efficacy"),
                RulePattern(r"\befficacy (was )?(shown|demonstrated|established)\b", "efficacy demonstrated"),
                RulePattern(r"\bbenefit (was )?(shown|demonstrated|established)\b", "benefit demonstrated"),
                RulePattern(r"\bmet (the )?(primary )?endpoint\b", "met endpoint"),
                RulePattern(r"\bsuperior(ity)?\b", "superiority"),
            ),
        ),
        CategoryRule(
            "futility",
            (
                RulePattern(r"\bfutilit(y|ies)\b", "futility"),
                RulePattern(r"\bfutile\b", "futile"),
                RulePattern(r"\black of (clinical )?efficacy\b", "lack of efficacy"),
                RulePattern(r"\blake of efficacy\b", "lack of efficacy (typo)"),  # common typo
                RulePattern(r"\bno (clinical )?efficacy\b", "no efficacy"),
                RulePattern(r"\binsufficient efficacy\b", "insufficient efficacy"),
                RulePattern(r"\bineffective\b", "ineffective"),
                RulePattern(r"\bnot effective\b", "not effective"),
                RulePattern(r"\bno benefit\b", "no benefit"),
                RulePattern(r"\black of (clinical )?benefit\b", "lack of benefit"),
                RulePattern(r"\black of evidence of (clinical )?benefit\b", "lack of evidence of benefit"),
                RulePattern(r"\bconditional power\b", "conditional power"),
                RulePattern(r"\bfailed\b.{0,60}\b(endpoint|efficacy|interim)\b", "failed endpoint/efficacy"),
                RulePattern(r"\blow likelihood\b.{0,60}\b(achiev|efficacy|benefit|endpoint)\b", "low likelihood of success"),
                RulePattern(r"\bunlikely\b.{0,60}\b(achiev|efficacy|benefit|endpoint)\b", "unlikely to succeed"),
                # Published competing evidence / external data
                RulePattern(r"\bpublished.{0,80}\b(evidence|results|data|study|trial)\b", "published evidence"),
                RulePattern(r"\bsimilar.{0,40}\b(data|results|study|findings)\b.{0,40}\b(published|obtained|shown|available|resolved)\b", "similar data available"),
                RulePattern(r"\b(data|results|evidence|findings)\b.{0,40}\b(not support|do not support|does not support|don.t support)\b", "data does not support"),
                RulePattern(r"\bpreliminary (data|results).{0,60}\b(not support|do not support|does not support)\b", "preliminary data not support"),
                # Interim analysis showing futility
                RulePattern(r"\binterim analysis\b", "interim analysis"),
                RulePattern(r"\bplanned interim\b", "planned interim"),
                RulePattern(r"\binterim.{0,80}\b(worse|inferior|negative|no significant|failed|not significant|low|unable to show)\b", "interim analysis negative"),
                RulePattern(r"\b(odds|probability|likelihood).{0,40}\b(proving|meeting|achieving).{0,40}\b(hypothesis|primary|endpoint)\b.{0,40}\blow\b", "low odds of success"),
                # Did not meet statistical/efficacy bar
                RulePattern(r"\bdid not meet.{0,40}\b(statistical|efficacy|endpoint|primary|benchmark)\b", "did not meet bar"),
                RulePattern(r"\black of.{0,30}\b(treatment |group |)difference\b", "lack of treatment difference"),
                RulePattern(r"\bbenchmark (efficacy|endpoint|criteria)\b", "benchmark efficacy"),
                RulePattern(r"\bdid not (advance|progress|pass|reach phase)\b", "did not advance"),
                RulePattern(r"\bnot advance.{0,20}\b(to |phase)\b", "did not advance to phase"),
                RulePattern(r"\bhas not demonstrated efficacy\b", "has not demonstrated efficacy"),
                # Improvements in standard of care rendering trial unnecessary
                RulePattern(r"\bimprovements? in (clinical practice|standard of care|current (therapy|treatment))\b", "SoC improvement"),
                RulePattern(r"\b(reduced|reduce).{0,40}\b(apparent |)incidence\b", "reduced incidence"),
                RulePattern(r"\b(resolved?|address).{0,30}\b(study goal|research question|question this study)\b", "study goal resolved"),
                RulePattern(r"\bnew data\b.{0,60}\b(not support|do not support|does not|no longer)\b", "new data does not support"),
                RulePattern(r"\b(primary |)endpoint not met\b", "endpoint not met"),
                RulePattern(r"\bnot met\b.{0,20}\b(primary|endpoint|goal|hypothesis)\b", "endpoint not met (reversed)"),
                RulePattern(r"\black of.{0,30}\b(primary outcome|primary endpoint|primary efficacy)\b", "lack of primary efficacy"),
                RulePattern(r"\b(recent |new )(research|data|evidence|literature|study|trial|results)\b.{0,80}\b(effective|efficacious|beneficial|demonstrated|shown|indicat)\b", "recent evidence shows efficacy"),
                RulePattern(r"\bpilot.{0,60}\b(satisfying|conclusive|positive|sufficient|resolv)\b", "pilot study conclusive"),
                RulePattern(r"\bsatisfying.{0,40}\bpilot\b", "pilot satisfying (reversed)"),
                RulePattern(r"\bphase (i|1|ii|2)\b.{0,60}\b(not support|did not advance|did not support|did not progress)\b", "phase did not advance"),
                RulePattern(r"\bresults.{0,60}\b(resolv|address).{0,30}\bgoal\b", "results resolved goal"),
                RulePattern(r"\bgood.{0,20}\b(result|outcome|finding)\b.{0,60}\bpilot\b", "good pilot results"),
                RulePattern(r"\bno difference in outcome\b", "no difference in outcome"),
                RulePattern(r"\bno significant (difference|effect|benefit|improvement)\b", "no significant difference"),
                RulePattern(r"\bpoor efficacy\b", "poor efficacy"),
                RulePattern(r"\blacking (effect|efficacy|benefit)\b", "lacking effect"),
                RulePattern(r"\b(didn.t|did not|does not|don.t|no longer) show any benefit\b", "no benefit shown"),
                RulePattern(r"\b(performance|success).{0,30}\b(insufficient|inadequate|poor|below.{0,10}threshold)\b", "insufficient performance"),
                RulePattern(r"\bno longer consistent with (current |)clinical practice\b", "no longer consistent with SoC"),
                RulePattern(r"\b(licenced|licensed|approved).{0,60}\b(no longer necessary|not necessary|not needed)\b", "licensed making study unnecessary"),
                RulePattern(r"\bstop\w*.{0,40}\bbased on efficacy result\b", "stopped based on efficacy results"),
            ),
        ),
        CategoryRule(
            "recruitment_failure",
            (
                RulePattern(r"\brecruit(ment|ing|ed|er)?\b", "recruitment"),
                RulePattern(r"\brecruitement\b", "recruitement (typo)"),   # French-influenced
                RulePattern(r"\brecrueted\b", "recrueted (typo)"),
                RulePattern(r"\brecruited\b", "recruited"),
                RulePattern(r"\brecruiting\b", "recruiting"),
                RulePattern(r"\benroll(ment|ing|ed)?\b", "enrollment"),
                RulePattern(r"\benrol(ment|ling|led|l)?\b", "enrolment (British)"),  # single-l British spelling
                RulePattern(r"\benrollement\b", "enrollement (typo)"),
                RulePattern(r"\benrole\b", "enrole (typo)"),
                RulePattern(r"\baccrual\b", "accrual"),
                RulePattern(r"\baccural\b", "accural (typo)"),
                RulePattern(r"\bnon[- ]accrual\b", "non-accrual"),
                RulePattern(r"\binclusion rate\b", "inclusion rate"),
                RulePattern(r"\bpatient population\b", "patient population"),
                # Lack-of-patients phrases (many variants)
                RulePattern(r"\black of (eligible |suitable |enough )?(patients?|subjects?|participants?|volunteers?)\b", "lack of patients"),
                RulePattern(r"\black of inclusion\b", "lack of inclusion"),
                RulePattern(r"\bno (patients?|subjects?|participants?) (were |have been |)(recruited|enrolled|accrued)\b", "no patients recruited"),
                RulePattern(r"\bno patients? recruited\b", "no patients recruited"),
                RulePattern(r"\bnot (enough|sufficient).{0,30}\b(patients?|subjects?|participants?|volunteers?)\b", "not enough patients"),
                RulePattern(r"\bnot enough.{0,30}\b(recruited|enrolled|accrued)\b", "not enough enrolled"),
                RulePattern(r"\binability to enro(l+)\b", "inability to enroll"),
                RulePattern(r"\bunable to (enro(l+)|accrue)\b", "unable to enroll"),
                RulePattern(r"\bnever accrued\b", "never accrued"),
                RulePattern(r"\binsufficient (patients?|subjects?|participants?|sample|accrual|number)\b", "insufficient patients"),
                RulePattern(r"\beligible (patients?|subjects?|participants?|population)\b", "eligible patients"),
                RulePattern(r"\bpatient sample not reached\b", "patient sample not reached"),
                RulePattern(r"\bslow.{0,30}\b(recruit|enroll|accrual)\b", "slow recruitment"),
                RulePattern(r"\bdifficult(ies|y)?.{0,40}\b(recruit|enroll|accrual)\b", "difficult recruitment"),
                RulePattern(r"\bno (patients?|subjects?|participants?) willing\b", "no patients willing"),
                RulePattern(r"\bnot finding.{0,30}\b(patients?|subjects?|participants?)\b", "not finding patients"),
                RulePattern(r"\b(finding|find).{0,30}\b(patients?|subjects?)\b.{0,30}\b(includ|enrol|recruit)\b", "finding patients"),
                RulePattern(r"\bpatient (availability|interest)\b", "patient availability"),
                RulePattern(r"\bscarce interest of.{0,30}\b(centres?|centers?|sites?)\b", "scarce centre interest"),
                RulePattern(r"\bunable to (generate|find|get|identify).{0,30}\b(patients?|subjects?|participants?)\b", "unable to get patients"),
                RulePattern(r"\b(patients?|subjects?|participants?).{0,30}\bnot (entered|recruited|enrolled)\b", "patients not entered"),
                # Withdrawal of participants
                RulePattern(r"\bparticipants? withdrawal\b", "participant withdrawal"),
                RulePattern(r"\bwithdrawn? due to lack of (eligible|suitable|willing)?\b", "withdrawn lack of patients"),
                RulePattern(r"\bdifficulties? of.{0,20}\b(recruit|enrol|enroll)\w*\b", "difficulties recruiting"),
                RulePattern(r"\bproblems? with.{0,30}\b(enroll|enrol|recruit)\w*\b", "problems enrolling"),
                RulePattern(r"\bslow.{0,30}\b(enrol|enroll)\w*\b", "slow enrollment"),
                RulePattern(r"\blow.{0,30}\b(enrol|enroll)\w*\brate\b", "low enrollment rate"),
                RulePattern(r"\blow.{0,10}\benrol(l?)ment\b", "low enrolment"),
                RulePattern(r"\bless (patients?|subjects?|participants?|cases?).{0,20}than expected\b", "less patients than expected"),
                RulePattern(r"\bno patients? (have been |were |are |)recruited\b", "no patients recruited (variant)"),
                RulePattern(r"\bno (subjects?|participants?|volunteers?).{0,20}(recruited|enrolled|accrued)\b", "no subjects enrolled"),
                RulePattern(r"\babsence of inclusion (criteria)?\b", "absence of inclusion criteria"),
                RulePattern(r"\bdifficult\w*.{0,20}(to |in )?(include|includ\w*)\b", "difficulty including patients"),
                RulePattern(r"\bnot enough.{0,30}\binclusion\w*\b", "not enough inclusions"),
                RulePattern(r"\bnumber of.{0,50}\b(participants?|subjects?|patients?)\b.{0,40}\bnot (obtain\w*|reach\w*|achiev\w*)\b", "patient number not obtained"),
                RulePattern(r"\b(investigators?|researchers?).{0,60}\b(could not|cannot|couldn.t) (identify|find|locate).{0,30}\b(subjects?|participants?|patients?)\b", "could not identify subjects"),
                RulePattern(r"\binclusion curve\b", "inclusion curve"),
                RulePattern(r"\black of availability.{0,30}\b(participants?|subjects?|patients?)\b", "lack of patient availability"),
                RulePattern(r"\battrition.{0,30}\b(greater|higher|unexpected|high)\b", "high attrition"),
            ),
        ),
        CategoryRule(
            "business_decision",
            (
                RulePattern(r"\bbusiness\b", "business"),
                RulePattern(r"\bcommercial\b", "commercial"),
                RulePattern(r"\bstrategic\b", "strategic"),
                RulePattern(r"\bportfolio\b", "portfolio"),
                RulePattern(r"\bcompany decision\b", "company decision"),
                RulePattern(r"\bsponsor decision\b", "sponsor decision"),
                RulePattern(r"\bcompany acquired\b", "company acquired"),
                RulePattern(r"\bacqui(red|sition)\b", "acquisition"),
                RulePattern(r"\bdissolv(ed)?\b", "company dissolved"),
                RulePattern(r"\bproduct discontinued\b", "product discontinued"),
                RulePattern(r"\bdevelopment (program|plan)\b", "development program"),
                RulePattern(r"\bmarketing\b", "marketing"),
                # Sponsor/company actions
                RulePattern(r"\bsponsor.{0,40}\b(withdrew|withdraw|decided|terminat|not.{0,20}proceed|not.{0,20}interest|not.{0,20}pursu)\b", "sponsor action"),
                RulePattern(r"\bwithdrawal of (the |)support\b", "withdrawal of support"),
                RulePattern(r"\bwithdr(ew|awn) (the |)support\b", "withdrew support"),
                RulePattern(r"\b(terminated|closed|cancelled|withdrawn) by (the )?sponsor\b", "by sponsor"),
                RulePattern(r"\bsponsors?.{0,20}\b(has |have |)(decided|chosen|concluded|determined)\b", "sponsor decided"),
                RulePattern(r"\bcompany.{0,40}\b(decided|chosen|concluded|no longer|discontinu)\b", "company decided"),
                RulePattern(r"\binvestigators?.{0,10}decision\b", "investigator decision"),
                RulePattern(r"\buniversity decision\b", "university decision"),
                RulePattern(r"\binstitution.{0,10}decision\b", "institution decision"),
                # Strategic / refocus language
                RulePattern(r"\bclinical strategy\b", "clinical strategy"),
                RulePattern(r"\bprogram refocus\b", "program refocus"),
                RulePattern(r"\bchange in (research|clinical|development|study|program|company).{0,20}\b(focus|direction|strategy|plan|priorities)\b", "change in strategy"),
                RulePattern(r"\bpriorities\b", "priorities"),
                RulePattern(r"\bnot (interested|planning).{0,20}\b(proceed|continue|pursue)\b", "not interested proceeding"),
                RulePattern(r"\bno longer pursu\w*\b", "no longer pursuing"),
                RulePattern(r"\bno longer (interested|planning|required|needed|necessary|viable|developing)\b", "no longer interested"),
                RulePattern(r"\bnot.{0,20}pursu(e|ing|ed)\b", "not pursuing"),
                RulePattern(r"\bnot proceed(ing)? with\b", "not proceeding"),
                RulePattern(r"\bfocusing on different\b", "focusing on different"),
                RulePattern(r"\brestructur\w*\b", "restructuring"),
                RulePattern(r"\bdiscontinue.{0,10}development\b", "discontinued development"),
                RulePattern(r"\bdevelopment.{0,30}discontinu\b", "development discontinued"),
                RulePattern(r"\bdid not (advance to|progress to|support).{0,30}\b(development|phase)\b", "did not advance to development"),
                RulePattern(r"\bnot going (to|forward)\b", "not going forward"),
                RulePattern(r"\bdecision not to (go|proceed|continue|pursue)\b", "decision not to proceed"),
                RulePattern(r"\bdecided not to\b", "decided not to"),
                RulePattern(r"\bsponsors? decision\b", "sponsors decision"),
                RulePattern(r"\bpi\b.{0,40}\b(not|did not|didn.t).{0,30}\b(go forward|proceed|continue|want to)\b", "PI decided not to proceed"),
                RulePattern(r"\bchosen.{0,60}\bnot to (continue|proceed|pursue)\b", "chosen not to proceed"),
                RulePattern(r"\bno longer deemed necessary\b", "no longer deemed necessary"),
                RulePattern(r"\bno longer practical\b", "no longer practical"),
                RulePattern(r"\bproject (has ended|ended|end)\b", "project ended"),
                RulePattern(r"\bproject.{0,20}\bstatus\b", "project status"),
                RulePattern(r"\bhas chosen.{0,40}\bnot to\b", "has chosen not to"),
                RulePattern(r"\bwill not (be )?conduct\w*\b", "will not conduct"),
                RulePattern(r"\bnot going forward\b", "not going forward"),
                RulePattern(r"\bdecision of.{0,20}\b(study|trial|principal|lead|primary) (investigator|pi)\b", "PI decision"),
                RulePattern(r"\bnot.*interested.{0,20}\b(study|trial|research)\b", "not interested in study"),
                RulePattern(r"\bdiscontinuation of (clinical investigation|development|the study)\b", "discontinuation of development"),
                RulePattern(r"\bdecision to (terminate|stop|discontinue).{0,30}\bdevelopment\b", "decision to terminate development"),
                RulePattern(r"\bstrategy change\b", "strategy change"),
                RulePattern(r"\bsponsor.{0,30}\b(halt\w*|terminat\w*)\b", "sponsor halted/terminated"),
                RulePattern(r"\bmanagement decision\b", "management decision"),
                RulePattern(r"\bwithdr(ew|awn).{0,30}\binterest\b", "withdrew interest"),
                RulePattern(r"\blost sponsorship\b", "lost sponsorship"),
                RulePattern(r"\bwithout (sufficient |)interest\b", "without interest"),
            ),
        ),
        CategoryRule(
            "administrative",
            (
                RulePattern(r"\bcovid[- ]?19\b", "COVID-19"),
                RulePattern(r"\bcovid\b", "COVID"),
                RulePattern(r"\bpandemic\b", "pandemic"),
                RulePattern(r"\bfunding\b", "funding"),
                RulePattern(r"\bstaff(ing)?\b", "staffing"),
                RulePattern(r"\bpersonnel\b", "personnel"),
                RulePattern(r"\bresources?\b", "resources"),
                RulePattern(r"\bpi leaving\b", "PI leaving"),
                RulePattern(r"\binvestigator left\b", "investigator left"),
                RulePattern(r"\bleft the institution\b", "left institution"),
                RulePattern(r"\binstitution\b", "institution"),
                RulePattern(r"\birb\w*\b", "IRB"),  # catches IRB, IRBO, IRBA etc.
                RulePattern(r"\bregulatory\b", "regulatory"),
                RulePattern(r"\bind\b", "IND"),
                RulePattern(r"\bprotocol\b", "protocol"),
                RulePattern(r"\blogistic(s|al)?\b", "logistics"),
                RulePattern(r"\btechnical\b", "technical"),
                RulePattern(r"\blapse of renewal\b", "lapse of renewal"),
                RulePattern(r"\bdata quality\b", "data quality"),
                RulePattern(r"\bmeasurement\b", "measurement"),
                RulePattern(r"\bshould not have been listed\b", "listed in error"),
                # Investigator / PI departure (many phrasings)
                RulePattern(r"\b(pi|principal investigator|primary investigator|co[- ]investigator)\b.{0,80}\b(left|moved|relocated|retired|departed|resigned|withdrew|leaving)\b", "PI/investigator left"),
                RulePattern(r"\b(left|moved|relocated|retired|departed|resigned)\b.{0,80}\b(pi|principal investigator|primary investigator|investigator)\b", "PI left (reversed)"),
                RulePattern(r"\bdeparture of.{0,60}\b(investigator|pi\b|co[- ]investigator)\b", "departure of investigator"),
                RulePattern(r"\binvestigator.{0,40}\b(left|moved|relocated|retired|departed|resigned|withdrew)\b", "investigator left"),
                RulePattern(r"\bpi (has |)(left|moved|relocated|retired|resigned|departed)\b", "PI left"),
                RulePattern(r"\bpi.{0,10}resigned\b", "PI resigned"),
                RulePattern(r"\bpi.{0,10}no longer\b", "PI no longer"),
                RulePattern(r"\b(retired|relocated|resigned).{0,60}\b(investigator|researcher|pi)\b", "investigator retired"),
                # Drug / product supply
                RulePattern(r"\bdrug supply\b", "drug supply"),
                RulePattern(r"\bsupply (issue|problem|shortage)\b", "supply issue"),
                RulePattern(r"\bimp\b", "IMP"),
                RulePattern(r"\binvestigational (medicinal )?product\b", "investigational product"),
                RulePattern(r"\bunavailability of\b", "unavailability"),
                RulePattern(r"\bdrug.{0,40}\b(discontinued|not available|unavailable)\b", "drug unavailable"),
                RulePattern(r"\bmanufacturer.{0,40}\b(unable|discontinued|stop|discontinu)\b", "manufacturer unable"),
                RulePattern(r"\b(drug|medication|medicine|product|compound|treatment).{0,30}\bnot available\b", "drug not available"),
                RulePattern(r"\bproduct.{0,20}\b(unavailable|discontinued)\b", "product unavailable"),
                RulePattern(r"\bproduction.{0,30}\b(issues?|problem|difficult|fail)\b", "production issues"),
                RulePattern(r"\bunable to (produce|manufacture|supply|get|obtain).{0,40}\b(drug|medication|product|compound|imp)\b", "unable to produce drug"),
                # Funding (unfunded / not funded)
                RulePattern(r"\bunfunded\b", "unfunded"),
                RulePattern(r"\bnot funded\b", "not funded"),
                RulePattern(r"\bgrant.{0,60}\b(not|was not|wasn.t|no|failed to).{0,20}\b(funded|approved|awarded)\b", "grant not funded"),
                RulePattern(r"\bfinancial.{0,30}\b(status|reason|shortage|constraint|problem|issue|difficul)\b", "financial reason"),
                RulePattern(r"\bfinancial problem\b", "financial problem"),
                RulePattern(r"\black of fund(ing|s|er)?\b", "lack of funding"),
                RulePattern(r"\black of support\b", "lack of support"),
                RulePattern(r"\bno.{0,10}\b(funding|funded)\b", "no funding"),
                RulePattern(r"\b(budget|budgetary).{0,30}\b(issue|problem|constraint|cut|reduction|insufficient|overrun)\b", "budget issue"),
                # Regulatory / approval failures
                RulePattern(r"\b(drug|national|competent|health|ethics).{0,20}authority\b", "authority"),
                RulePattern(r"\bec refused\b", "EC refused"),
                RulePattern(r"\bmhra\b", "MHRA"),
                RulePattern(r"\b(approval|authorisation|authorization).{0,40}\b(refused|denied|rejected|not granted|not obtained)\b", "approval refused"),
                RulePattern(r"\brefused (the |)(approval|authorisation)\b", "refused approval"),
                RulePattern(r"\b(did not|could not|unable to|failed to) (get|obtain|receive|gain|secure).{0,20}\b(approval|authorisation|clearance)\b", "did not get approval"),
                RulePattern(r"\bhealth authority\b", "health authority"),
                RulePattern(r"\binsurance.{0,40}\b(refus|denied|unable|cannot|could not)\b", "insurance refused"),
                RulePattern(r"\bprivacy (legislation|law|regulation)\b", "privacy legislation"),
                # Never initiated / started
                RulePattern(r"\bnever (started|activated|initiated|opened|implemented|launched)\b", "never started"),
                RulePattern(r"\bnot (yet )?(started|initiated|opened|launched)\b", "not started"),
                RulePattern(r"\bcancelled before (active|start|enrollment|initiation)\b", "cancelled before active"),
                RulePattern(r"\b(study|trial).{0,20}\b(was not|never|not) initiated\b", "not initiated"),
                RulePattern(r"\bnot (yet )?initiat\w*\b", "not initiated"),
                RulePattern(r"\b(study|trial).{0,20}\bnever (opened|started|activated)\b", "never opened"),
                # Other administrative misc
                RulePattern(r"\badministratively closed\b", "administratively closed"),
                RulePattern(r"\badministrative (reason|delay|issue|closure|problem)\b", "administrative reason"),
                RulePattern(r"\bspecimen collection\b", "specimen collection"),
                RulePattern(r"\bchanges? in the (organisation|organization)\b", "organisational changes"),
                RulePattern(r"\bbioanalytical\b", "bioanalytical issue"),
                RulePattern(r"\bdata (accuracy|inaccurate|inaccuracy|integrity)\b", "data accuracy"),
                RulePattern(r"\bprolongation.{0,20}\b(rejected|refused|not approved)\b", "prolongation rejected"),
                RulePattern(r"\bsterilization\b", "sterilization"),
                RulePattern(r"\bduplicate record\b", "duplicate record"),
                RulePattern(r"\bduplicate.{0,20}\b(study|trial|registration)\b", "duplicate study"),
                # PI departure (additional phrasings missed in first pass)
                RulePattern(r"\b(doctor|physician|researcher|clinician|fellow|resident)\b.{0,80}\b(left|moved|changed|relocated|no longer|left)\b", "staff departed"),
                RulePattern(r"\bdr\..{0,80}\b(left|moved|departed|relocated|resigned|no longer)\b", "Dr. departed"),
                RulePattern(r"\b(primary|principal|lead) investigator\b.{0,60}\bno longer\b", "PI no longer"),
                RulePattern(r"\bpi\b.{0,60}\b(unable to continue|no longer a part|no longer involved)\b", "PI unable to continue"),
                RulePattern(r"\b(lead |primary |principal )(researcher|investigator|scientist)\b.{0,60}\b(left|moved|completed|retired|resigned)\b", "lead researcher left"),
                RulePattern(r"\bresident\b.{0,40}\b(left|moved|completed|relocated)\b", "resident left"),
                # Equipment / technology unavailability
                RulePattern(r"\bequipment\b.{0,40}\b(not functional|fail\w*|broke\w*|malfunction\w*|not available|unavailable)\b", "equipment failure"),
                RulePattern(r"\b(no longer offered|no longer available|no longer practical)\b", "no longer available/offered"),
                RulePattern(r"\b(technology|technique|procedure|method)\b.{0,40}\bno longer\b", "technology no longer available"),
                RulePattern(r"\badministrative delay\w*\b", "administrative delays"),
                RulePattern(r"\blocal regulation\w*\b", "local regulations"),
                RulePattern(r"\bmedical device law\b", "medical device law"),
                RulePattern(r"\bsecretariat of health\b", "secretariat of health"),
                RulePattern(r"\binsurance\b.{0,60}\b(refusal|refused|rejection|denied|unable|not obtain)\b", "insurance refused"),
                RulePattern(r"\bfaillure to obtain\b", "failure to obtain (typo)"),
                RulePattern(r"\bfeasibilit\w*\b.{0,40}\bnot met\b", "feasibility not met"),
                RulePattern(r"\bnot feasible\b", "not feasible"),
                RulePattern(r"\bmanufacturer.{0,40}\b(ceased|stopped|discontinu\w*).{0,40}\b(produce|production|manufactur)\b", "manufacturer ceased"),
                RulePattern(r"\bceased to (produce|manufactur|supply)\b", "ceased to produce"),
                RulePattern(r"\b(vaccine|drug|medication|compound|product).{0,30}\bexpired\b", "drug expired"),
                RulePattern(r"\binoperable\b", "inoperable"),
                RulePattern(r"\badministrative reason\w*\b", "administrative reasons"),
                RulePattern(r"\bnot human subject\b", "not human subject research"),
                RulePattern(r"\badministratively complete\w*\b", "administratively complete"),
                RulePattern(r"\bclinical hold\b", "clinical hold"),
                RulePattern(r"\bfda\b.{0,40}\b(hold|approval|issue|action|request|response)\b", "FDA action"),
                RulePattern(r"\bsoftware (issue|problem|error|fail\w*)\b", "software issue"),
                RulePattern(r"\bdevice.{0,40}\b(not approved|not cleared|unapproved)\b", "device not approved"),
                RulePattern(r"\binvestigator no longer\b", "investigator no longer"),
                RulePattern(r"\beuropean.{0,30}\b(drug |)(approval|authorisation|authorization|regulatory)\b", "European regulatory"),
            ),
        ),
    )

    # Cache for combined compiled regexes — built once at first call.
    _COMBINED_REGEX: Optional[Dict[str, re.Pattern[str]]] = None

    UNCLEAR_PATTERNS: Tuple[RulePattern, ...] = (
        RulePattern(r"^\s*(study )?(cancelled|canceled|terminated|withdrawn)\.?\s*$", "generic cancellation"),
        RulePattern(r"\bsee .*detailed description\b", "see detailed description"),
        RulePattern(r"\bnot pursued\b", "not pursued"),
        # Committee recommendations without stated reason
        RulePattern(r"\b(dsmb|dmb|dsmc|dmc|data monitoring).{0,30}recommendation\b", "committee recommendation"),
        RulePattern(r"\brecommendation of (the )?(dsmb|dmb|dsmc|dmc|data monitoring|trial steering|safety)\b", "committee recommendation"),
        RulePattern(r"^\s*(dsmb|dmb|dsmc|dmc) recommendation\.?\s*$", "generic committee recommendation"),
    )

    # How many distinct categories must co-occur to label "mixed_reasons"
    MIXED_REASONS_THRESHOLD: int = 3

    @classmethod
    def _build_combined_regex(cls) -> Dict[str, re.Pattern[str]]:
        """Build one compiled combined regex per category plus one for unclear patterns."""
        def _nc(pattern_str: str) -> str:
            return re.sub(r"\((?!\?)", "(?:", pattern_str)

        result: Dict[str, re.Pattern[str]] = {}
        for rule in cls.RULES:
            joined = "|".join(f"(?:{_nc(p.regex)})" for p in rule.patterns)
            result[rule.category] = re.compile(joined, re.IGNORECASE)
        unclear = "|".join(f"(?:{_nc(p.regex)})" for p in cls.UNCLEAR_PATTERNS)
        result["__unclear__"] = re.compile(unclear, re.IGNORECASE)
        return result

    @classmethod
    def _get_combined_regex(cls) -> Dict[str, re.Pattern[str]]:
        if cls._COMBINED_REGEX is None:
            cls._COMBINED_REGEX = cls._build_combined_regex()
        return cls._COMBINED_REGEX

    # Maximum parallel workers for LLM fallback calls
    LLM_MAX_WORKERS: int = 30

    # Context columns forwarded to the LLM prompt (mirrors _compact_context)
    _LLM_CONTEXT_COLS: Tuple[str, ...] = (
        "brief_title",
        "official_title",
        "conditions",
        "intervention_names",
        "primary_outcomes",
        "secondary_outcomes",
    )

    def classify_dataframe(
        self,
        text_series: pd.Series,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Vectorised rule-based classification with LLM fallback for ambiguous rows.

        All rows go through the rule-based pass (str.contains with pre-compiled
        combined regexes). Rows flagged needs_llm=True are then sent to the LLM
        in parallel — same pattern as the paper: LLM adjudicates ambiguous,
        mixed, uncertain, or low-confidence termination reasons using the
        prespecified category schema.

        Parameters
        ----------
        text_series : why_stopped_text column aligned to the trial DataFrame.
        df : full trial DataFrame; supplies trial context for LLM prompts.
        """
        import hashlib as _hashlib
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac

        idx = text_series.index

        norm = text_series.fillna("").astype(str).str.strip()
        norm = norm.where(~norm.str.lower().isin({"nan", "none", "null"}), "")
        norm = norm.str.replace(r"\s+", " ", regex=True)
        has_text = norm.str.len() > 0

        rx = self._get_combined_regex()
        cat_order = [r.category for r in self.RULES]

        hit_df = pd.DataFrame(
            {cat: norm.str.contains(rx[cat], na=False) for cat in cat_order},
            index=idx,
        )
        unclear_hit = norm.str.contains(rx["__unclear__"], na=False)

        hit_counts = hit_df[cat_order].sum(axis=1)
        no_hit = hit_counts == 0

        # Assign primary: iterate from lowest to highest priority so highest wins
        primary = pd.Series("uncertain", index=idx, dtype=object)
        for cat in reversed(self.CATEGORY_PRIORITY):
            primary = primary.where(~hit_df[cat], cat)

        # 3+ distinct categories → mixed_reasons
        mixed = (hit_counts >= self.MIXED_REASONS_THRESHOLD) & has_text
        primary = primary.where(~mixed, "mixed_reasons")

        # No hits + unclear signal → "unclear"
        primary = primary.where(~(no_hit & unclear_hit & has_text), "unclear")
        # No text → always "unclear"
        primary = primary.where(has_text, "unclear")

        is_scientific = primary.isin(self.SCIENTIFIC_CATEGORIES)
        multi_hit = hit_counts > 1

        confidence = pd.Series(
            np.select(
                [
                    ~has_text,
                    no_hit & unclear_hit & has_text,
                    no_hit & ~unclear_hit & has_text,
                    multi_hit & is_scientific,
                    multi_hit & ~is_scientific,
                    ~multi_hit & is_scientific,
                ],
                [0.0, 0.2, 0.35, 0.78, 0.72, 0.92],
                default=0.88,
            ),
            index=idx,
            dtype=float,
        )

        # "unclear" never needs LLM — text is uninformative.
        # "uncertain" and "mixed_reasons" do; low-confidence / multi-hit also do.
        needs_llm = (
            primary.isin({"uncertain", "mixed_reasons"})
            | (confidence < self.llm_confidence_threshold)
            | multi_hit
        ) & has_text & ~primary.eq("unclear")

        method = pd.Series("rule", index=idx, dtype=object).where(has_text, "missing")
        rationale = pd.Series("", index=idx, dtype=object)

        # ── LLM fallback (paper §2.6: adjudicate ambiguous / uncertain rows) ─
        if self.llm_client:
            llm_idx_list = idx[needs_llm].tolist()
            if llm_idx_list:
                # Pre-extract text + context before launching threads
                _text_data: Dict[Any, str] = norm.loc[llm_idx_list].to_dict()
                _ctx_cols = [c for c in self._LLM_CONTEXT_COLS
                             if df is not None and c in df.columns]
                _ctx_data: Dict[Any, Dict[str, Any]] = (
                    df.loc[llm_idx_list, _ctx_cols].to_dict("index")
                    if _ctx_cols else {}
                )
                _status_data: Dict[Any, Optional[str]] = (
                    df.loc[llm_idx_list, "overall_status"].to_dict()
                    if df is not None and "overall_status" in df.columns else {}
                )

                _cache: Dict[str, Optional[Dict[str, Any]]] = {}

                def _call_one(row_idx: Any) -> tuple:
                    text = _text_data[row_idx]
                    status = _status_data.get(row_idx)
                    raw_ctx = _ctx_data.get(row_idx, {})
                    context = {
                        k: v for k, v in raw_ctx.items()
                        if v is not None
                        and not (isinstance(v, float) and np.isnan(v))
                        and str(v).strip() not in {"", "nan", "None"}
                    }
                    fp = _hashlib.md5(
                        (text + str(status) + str(sorted(context.items()))).encode()
                    ).hexdigest()
                    if fp in _cache:
                        return row_idx, _cache[fp]
                    messages = self.build_llm_messages(text, status, context or None)
                    try:
                        raw = self.llm_client.classify(messages)
                    except Exception:
                        _cache[fp] = None
                        return row_idx, None
                    cat = str(raw.get("termination_category", "")).strip().lower()
                    if cat not in TERMINATION_CATEGORIES:
                        _cache[fp] = None
                        return row_idx, None
                    conf = self._clamp_confidence(raw.get("confidence", 0.0))
                    rat = str(raw.get("rationale", "")).strip()
                    res: Dict[str, Any] = {"category": cat, "confidence": conf, "rationale": rat}
                    _cache[fp] = res
                    return row_idx, res

                n_workers = min(self.LLM_MAX_WORKERS, len(llm_idx_list))
                llm_updates: Dict[Any, Dict[str, Any]] = {}
                with _TPE(max_workers=n_workers) as pool:
                    for fut in _ac(pool.submit(_call_one, i) for i in llm_idx_list):
                        row_idx, res = fut.result()
                        if res is not None and res["confidence"] >= self.llm_confidence_threshold:
                            llm_updates[row_idx] = res

                for row_idx, res in llm_updates.items():
                    primary.at[row_idx] = res["category"]
                    confidence.at[row_idx] = res["confidence"]
                    method.at[row_idx] = "llm"
                    needs_llm.at[row_idx] = False
                    rationale.at[row_idx] = res["rationale"]

        return pd.DataFrame(
            {
                "termination_category": primary,
                "termination_confidence": confidence.round(3),
                "termination_classification_method": method,
                "termination_matched_terms": pd.Series("", index=idx),
                "termination_secondary_categories": pd.Series("", index=idx),
                "termination_needs_llm": needs_llm,
                "termination_rationale": rationale,
            },
            index=idx,
        )

    def __init__(
        self,
        llm_client: Optional[TerminationLLMClient] = None,
        llm_confidence_threshold: float = 0.75,
    ) -> None:
        self.llm_client = llm_client
        self.llm_confidence_threshold = float(llm_confidence_threshold)
        self._llm_cache: Dict[str, TerminationClassification] = {}

    def classify(
        self,
        reason_text: Any,
        overall_status: Optional[str] = None,
        trial_context: Optional[Mapping[str, Any]] = None,
    ) -> TerminationClassification:
        """Classify a single whyStopped value."""
        rule_result = self.classify_with_rules(reason_text)

        if not (self.llm_client and rule_result.needs_llm):
            return rule_result

        text = self._normalize_text(reason_text)
        cache_key = self._cache_key(text, overall_status, trial_context)
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        try:
            llm_result = self._classify_with_llm(
                text=text,
                overall_status=overall_status,
                trial_context=trial_context,
                rule_result=rule_result,
            )
        except Exception as exc:
            logger.warning("LLM termination classification failed: %s", exc)
            return rule_result

        self._llm_cache[cache_key] = llm_result
        return llm_result

    def classify_with_rules(self, reason_text: Any) -> TerminationClassification:
        """Run deterministic local rules only."""
        text = self._normalize_text(reason_text)
        if not text:
            return TerminationClassification(
                termination_category="unclear",
                confidence=0.0,
                method="missing",
                matched_terms=[],
                secondary_categories=[],
                needs_llm=False,
                rationale="No whyStopped text.",
            )

        hits = self._find_hits(text)
        if not hits:
            unclear_terms = self._find_unclear_terms(text)
            if unclear_terms:
                return TerminationClassification(
                    termination_category="unclear",
                    confidence=0.2,
                    method="rule",
                    matched_terms=unclear_terms,
                    secondary_categories=[],
                    needs_llm=False,
                    rationale="Text is too generic to classify from whyStopped alone.",
                )

            return TerminationClassification(
                termination_category="uncertain",
                confidence=0.35,
                method="rule",
                matched_terms=[],
                secondary_categories=[],
                needs_llm=True,
                rationale="No deterministic rule matched; text may contain a specific reason.",
            )

        primary = self._choose_primary_category(hits)
        secondary = [cat for cat in self.CATEGORY_PRIORITY if cat in hits and cat != primary]
        matched_terms = self._flatten_terms(hits)
        confidence = self._confidence(primary, hits)
        needs_llm = (
            primary in self.FALLBACK_LLM_CATEGORIES
            or confidence < self.llm_confidence_threshold
            or len(hits) > 1
        )

        return TerminationClassification(
            termination_category=primary,
            confidence=confidence,
            method="rule",
            matched_terms=matched_terms,
            secondary_categories=secondary,
            needs_llm=needs_llm,
            rationale=self._rule_rationale(primary, secondary),
        )

    @classmethod
    def build_llm_messages(
        cls,
        reason_text: str,
        overall_status: Optional[str] = None,
        trial_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Mapping[str, str]]:
        """Build a strict prompt for an LLM fallback classifier."""
        context = cls._compact_context(trial_context)
        system = (
            "You classify ClinicalTrials.gov whyStopped text into exactly one "
            "predefined termination category. Use only the supplied text and "
            "context. Do not infer facts from outside knowledge. Return JSON only."
        )
        user_payload = {
            "valid_categories": list(TERMINATION_CATEGORIES),
            "category_definitions": {
                "safety": "Stopped due to participant safety concerns, adverse events, toxicity, mortality, harm, risk-benefit imbalance, or a safety-driven DSMB/DMC stop.",
                "efficacy": "Stopped early because efficacy or benefit was clearly demonstrated (e.g., met primary endpoint, overwhelming efficacy).",
                "futility": "Stopped because achieving benefit was unlikely — includes interim analysis showing futility, lack of efficacy, published evidence resolving the question, or improvements in standard of care.",
                "administrative": "Stopped for operational, regulatory, funding, staffing, drug supply, PI departure, COVID, protocol, or institutional reasons.",
                "recruitment_failure": "Stopped because enrollment, accrual, or finding eligible participants failed — regardless of whether the sponsor then formally withdrew.",
                "business_decision": "Stopped for commercial, strategic, portfolio, acquisition, or sponsor/company decision reasons, where no scientific or operational driver dominates.",
                "mixed_reasons": "Multiple distinct stopping reasons are explicitly present (e.g., both recruitment failure and business decision) with no single dominant reason.",
                "uncertain": "A specific reason is stated but is too ambiguous, brief, or unusual to confidently assign to any category above.",
                "unclear": "Text is missing, circular, purely generic ('study terminated'), or refers elsewhere without stating a reason.",
            },
            "decision_rules": [
                "Choose one category only.",
                "If safety, efficacy, or futility is explicitly a driver, prefer it over operational or business context.",
                "If recruitment is the explicit driver, choose recruitment_failure even if a sponsor later formally withdrew.",
                "Use business_decision only when the reason is commercial or strategic and no scientific, operational, or recruitment driver dominates.",
                "Use mixed_reasons only when two or more distinct high-level reasons are stated with roughly equal weight.",
                "Use uncertain when a reason is given but does not fit any named category — do not use unclear unless text is truly uninformative.",
                "Set confidence from 0 to 1.",
                "Keep rationale under 20 words.",
            ],
            "overall_status": overall_status or "",
            "why_stopped": reason_text,
            "trial_context": context,
            "required_json_shape": {
                "termination_category": "one valid category",
                "confidence": "number from 0 to 1",
                "rationale": "brief reason",
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ]

    def _classify_with_llm(
        self,
        text: str,
        overall_status: Optional[str],
        trial_context: Optional[Mapping[str, Any]],
        rule_result: TerminationClassification,
    ) -> TerminationClassification:
        if not self.llm_client:
            return rule_result

        messages = self.build_llm_messages(text, overall_status, trial_context)
        raw = self.llm_client.classify(messages)
        category = str(raw.get("termination_category", raw.get("category", ""))).strip().lower()
        if category not in TERMINATION_CATEGORIES:
            return rule_result

        confidence = self._clamp_confidence(raw.get("confidence", rule_result.confidence))
        rationale = str(raw.get("rationale", "")).strip()
        return TerminationClassification(
            termination_category=category,
            confidence=confidence,
            method="llm",
            matched_terms=rule_result.matched_terms,
            secondary_categories=rule_result.secondary_categories,
            needs_llm=False,
            rationale=rationale,
        )

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            return ""
        return re.sub(r"\s+", " ", text)

    def _find_hits(self, text: str) -> Dict[str, List[str]]:
        hits: Dict[str, List[str]] = {}
        for rule in self.RULES:
            terms: List[str] = []
            for pattern in rule.patterns:
                if re.search(pattern.regex, text, flags=re.IGNORECASE):
                    terms.append(pattern.label)
            if terms:
                hits[rule.category] = sorted(set(terms))
        return hits

    def _find_unclear_terms(self, text: str) -> List[str]:
        return [
            pattern.label
            for pattern in self.UNCLEAR_PATTERNS
            if re.search(pattern.regex, text, flags=re.IGNORECASE)
        ]

    def _choose_primary_category(self, hits: Mapping[str, List[str]]) -> str:
        if len(hits) >= self.MIXED_REASONS_THRESHOLD:
            return "mixed_reasons"
        for category in self.CATEGORY_PRIORITY:
            if category in hits:
                return category
        return "uncertain"

    @staticmethod
    def _flatten_terms(hits: Mapping[str, List[str]]) -> List[str]:
        terms: List[str] = []
        for category, category_terms in hits.items():
            terms.extend(f"{category}:{term}" for term in category_terms)
        return sorted(set(terms))

    def _confidence(self, primary: str, hits: Mapping[str, List[str]]) -> float:
        if len(hits) > 1:
            return 0.78 if primary in self.SCIENTIFIC_CATEGORIES else 0.72
        return 0.92 if primary in self.SCIENTIFIC_CATEGORIES else 0.88

    @staticmethod
    def _rule_rationale(primary: str, secondary: Sequence[str]) -> str:
        if secondary:
            return f"Rule match for {primary}; mixed with {', '.join(secondary)}."
        return f"Rule match for {primary}."

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, confidence))

    @staticmethod
    def _compact_context(context: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        if not context:
            return {}
        allowed = (
            "brief_title",
            "official_title",
            "conditions",
            "intervention_names",
            "primary_outcomes",
            "secondary_outcomes",
        )
        compact: Dict[str, str] = {}
        for key in allowed:
            if key not in context:
                continue
            value = context[key]
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                text = "; ".join(str(v) for v in value if v is not None)
            else:
                text = str(value)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                compact[key] = text[:700]
        return compact

    @staticmethod
    def _cache_key(
        text: str,
        overall_status: Optional[str],
        trial_context: Optional[Mapping[str, Any]],
    ) -> str:
        payload = {
            "text": text,
            "overall_status": overall_status or "",
            "context": TerminationReasonClassifier._compact_context(trial_context),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def build_default_classifier(enable_llm: bool = False, model: str = "gpt-4o-mini") -> TerminationReasonClassifier:
    """
    Factory used by scripts.

    LLM use is opt-in because full-dataset classification can be slow and costly.
    """
    llm_client: Optional[TerminationLLMClient] = None
    if enable_llm:
        llm_client = OpenAITerminationLLMClient(model=model)
    return TerminationReasonClassifier(llm_client=llm_client)
