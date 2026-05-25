"""
Rule-based approximation of trials where DMC oversight is indicated by a
regulatory-risk proxy.

Three variables are produced per trial:

  dmc_expected        — PRIMARY tiered rule (primary paper variable).
                        Interventional + one of:
                          Tier 1 (standalone):  serious/life-threatening condition,
                                                serious clinical endpoint, or
                                                vulnerable age population.
                          Tier 2 (compound):    late phase (3/4) AND at least one of
                                                large enrollment, long duration,
                                                serious condition, serious endpoint,
                                                or regulated intervention.
                          Tier 3 (compound):    regulated intervention AND
                                                large enrollment OR long duration.
                        Phase 3/4 alone is a risk signal, not automatic evidence.
                        Regulated intervention alone is not automatic evidence.

  dmc_expected_strict — STRICT tiered rule (sensitivity analysis).
                        Requires BOTH a standalone Tier 1 indicator AND a compound
                        criterion (Tier 2 or Tier 3). Trials with only vulnerable
                        population or only late phase without additional compound
                        evidence are excluded.

  dmc_expected_broad  — BROAD single-feature rule (sensitivity analysis).
                        Interventional + any one guidance feature; mirrors the
                        original FDA/EMA checklist approach.

This module does not assert that a DMC was legally required. All three variables
are auditable approximations for research purposes.
"""

from __future__ import annotations

import ast
import json
import math
import re

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from pydantic import BaseModel as _PydanticBase, Field as _Field
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple


DMC_EXPECTED_REASONS: Tuple[str, ...] = (
    "late_phase",
    "serious_or_life_threatening_condition",
    "vulnerable_age_population",
    "large_enrollment",
    "regulated_intervention",
    "long_duration",
    "serious_endpoint",
)


REASON_LABELS: Dict[str, str] = {
    "late_phase": "Phase 3 or Phase 4",
    "serious_or_life_threatening_condition": "Serious/life-threatening condition",
    "vulnerable_age_population": "Child or elderly eligible population",
    "large_enrollment": "Large enrollment",
    "regulated_intervention": "Drug/biologic/device intervention",
    "long_duration": "Long study duration",
    "serious_endpoint": "Serious clinical endpoint",
}


@dataclass(frozen=True)
class DMCExpectedResult:
    """Structured result for one trial."""

    # Primary paper variable: strict tiered regulatory-risk proxy
    dmc_expected: bool
    dmc_expected_reason_count: int
    dmc_expected_reasons: List[str]
    dmc_expected_reason_labels: List[str]
    dmc_expected_rule: str
    dmc_expected_tier: str  # "tier1", "tier2", "tier3", or ""
    # Sensitivity analysis: strict rule requiring tier1 AND (tier2 OR tier3)
    dmc_expected_strict: bool
    # Sensitivity analysis: broad single-feature rule (original behaviour)
    dmc_expected_broad: bool
    is_interventional_trial: bool
    dmc_expected_late_phase: bool
    dmc_expected_serious_condition: bool
    dmc_expected_vulnerable_population: bool
    dmc_expected_large_enrollment: bool
    dmc_expected_regulated_intervention: bool
    dmc_expected_long_duration: bool
    dmc_expected_serious_endpoint: bool
    trial_duration_months: Optional[float]
    dmc_expected_llm_used: bool = False
    dmc_expected_llm_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dict for pandas output."""
        data = asdict(self)
        data["dmc_expected_reasons"] = "; ".join(self.dmc_expected_reasons)
        data["dmc_expected_reason_labels"] = "; ".join(self.dmc_expected_reason_labels)
        return data


class DMCExpectedLLMClient(Protocol):
    """Protocol for an optional LLM provider that can infer DMC guidance features."""

    def classify(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        """Return JSON-like output with serious condition and serious endpoint flags."""


class OpenAIDMCExpectedLLMClient:
    """Optional OpenAI adapter for DMC expected inference."""

    RESPONSE_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "serious_or_life_threatening_condition": {"type": "boolean"},
            "serious_endpoint": {"type": "boolean"},
            "confidence": {"type": "number", "description": "Confidence from 0 to 1."},
            "rationale": {"type": "string", "description": "Brief rationale under 20 words."},
        },
        "required": [
            "serious_or_life_threatening_condition",
            "serious_endpoint",
            "confidence",
            "rationale",
        ],
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
                        "name": "dmc_expected_reason_inference",
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
                    "name": "dmc_expected_reason_inference",
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


if _HAS_PYDANTIC:
    class DMCLLMResponse(_PydanticBase):
        serious_or_life_threatening_condition: bool
        serious_endpoint: bool
        confidence: float = _Field(ge=0.0, le=1.0)
        rationale: str


class DMCExpectedClassifier:
    """
    Classify trials as DMC-expected when interventional trials have guidance
    features associated with recommended independent monitoring.

    Defaults are intentionally explicit so sensitivity analyses can adjust them:
        - large enrollment: >= 500 participants
        - long duration: >= 24 months from start to completion or primary completion
        - elderly eligibility: explicit maximum age >= 65 years or minimum age >= 65
    """

    REGULATED_INTERVENTION_TYPES = {
        "BIOLOGICAL",
        "BIOLOGIC",
        "BIOLOGICAL_PRODUCT",
        "COMBINATION_PRODUCT",
        "DEVICE",
        "DRUG",
        "GENETIC",
        "RADIATION",
    }

    SERIOUS_CONDITION_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\b(cancer|carcinoma|neoplasm|tumou?r|leukemia|lymphoma|myeloma|melanoma|sarcoma|malignan\w*|metasta\w*)\b",
            r"\b(heart failure|myocardial infarction|acute coronary|cardiac arrest|coronary artery disease)\b",
            r"\b(stroke|cerebrovascular|intracranial hemorrhage|subarachnoid hemorrhage)\b",
            r"\b(sepsis|septic shock|shock|critical illness|intensive care|icu)\b",
            r"\b(hiv|aids|tuberculosis|malaria|hepatitis|covid-19|covid)\b",
            r"\b(respiratory failure|copd|pulmonary hypertension|acute respiratory distress)\b",
            r"\b(renal failure|kidney failure|end stage renal|liver failure|cirrhosis)\b",
            r"\b(amyotrophic lateral sclerosis|als|multiple sclerosis|parkinson|alzheimer)\b",
            r"\b(sickle cell|hemophilia|thrombo\w*|embol\w*)\b",
            r"\b(transplant|trauma|burn injury|organ failure)\b",
        )
    )

    SERIOUS_ENDPOINT_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\b(death|mortality|survival|overall survival|progression.free survival)\b",
            # "adverse event" alone is too broad; require SAE/DLT explicitly
            r"\b(serious adverse event\w*|SAE)\b",
            r"\bdose.limiting toxicit\w*\b",
            r"\b(hospitali[sz]ation|icu|intensive care|ventilation)\b",
            r"\b(stroke|myocardial infarction|major adverse cardiovascular|mace)\b",
            r"\b(transplant|renal failure|liver failure|respiratory failure)\b",
            r"\b(disability|disabling)\b",
        )
    )

    # Pre-compiled combined patterns for vectorised pandas str.contains.
    # All capturing groups are converted to non-capturing so pandas does not warn.
    @staticmethod
    def _non_capturing(pattern_str: str) -> str:
        return re.sub(r"\((?!\?)", "(?:", pattern_str)

    _SERIOUS_CONDITION_COMBINED: re.Pattern[str] = re.compile(
        "|".join(
            f"(?:{re.sub(r'[(](?![?])', '(?:', p.pattern)})"
            for p in SERIOUS_CONDITION_PATTERNS
        ),
        re.IGNORECASE,
    )
    _SERIOUS_ENDPOINT_COMBINED: re.Pattern[str] = re.compile(
        "|".join(
            f"(?:{re.sub(r'[(](?![?])', '(?:', p.pattern)})"
            for p in SERIOUS_ENDPOINT_PATTERNS
        ),
        re.IGNORECASE,
    )

    # Vectorised regulated-intervention check — word-boundary anchored per type
    _REGULATED_INTERVENTION_PATTERN: re.Pattern[str] = re.compile(
        "|".join(rf"\b{t}\b" for t in sorted(REGULATED_INTERVENTION_TYPES)),
        re.IGNORECASE,
    )

    # Maximum parallel workers for LLM calls
    LLM_MAX_WORKERS: int = 30

    def __init__(
        self,
        large_enrollment_threshold: int = 500,
        long_duration_months_threshold: float = 24.0,
        elderly_age_threshold: float = 65.0,
        llm_client: Optional[DMCExpectedLLMClient] = None,
        llm_confidence_threshold: float = 0.75,
    ) -> None:
        self.large_enrollment_threshold = int(large_enrollment_threshold)
        self.long_duration_months_threshold = float(long_duration_months_threshold)
        self.elderly_age_threshold = float(elderly_age_threshold)
        self.llm_client = llm_client
        self.llm_confidence_threshold = float(llm_confidence_threshold)

    def classify(self, row: Mapping[str, Any]) -> DMCExpectedResult:
        """Classify one row from the ClinicalTrials.gov analysis dataset."""
        is_interventional = self._is_interventional(row.get("study_type"))
        end_date = row.get("completion_date")
        if self._is_missing(end_date):
            end_date = row.get("primary_completion_date")
        duration_months = self._duration_months(
            row.get("start_date"),
            end_date,
        )

        flags = {
            "late_phase": self._is_late_phase(row.get("phase")),
            "serious_or_life_threatening_condition": self._has_serious_condition(row),
            "vulnerable_age_population": self._has_vulnerable_age(row),
            "large_enrollment": self._has_large_enrollment(row.get("enrollment")),
            "regulated_intervention": self._has_regulated_intervention(row.get("intervention_types")),
            "long_duration": (
                duration_months is not None
                and duration_months >= self.long_duration_months_threshold
            ),
            "serious_endpoint": self._has_serious_endpoint(row),
        }

        llm_used = False
        llm_rationale = ""
        if self.llm_client and is_interventional and (
            not flags["serious_or_life_threatening_condition"]
            and not flags["serious_endpoint"]
        ):
            llm_result = self._classify_with_llm(row, flags)
            if llm_result is not None:
                flags.update(llm_result["flags"])
                llm_used = True
                llm_rationale = llm_result.get("rationale", "")

        reasons = [reason for reason in DMC_EXPECTED_REASONS if flags[reason]]

        # ── Broad rule (sensitivity analysis): any single guidance feature ──────
        dmc_expected_broad = bool(is_interventional and reasons)

        # ── Strict tiered rule (primary paper variable) ─────────────────────────
        # Tier 1 — standalone strong indicators (each sufficient on its own)
        tier1 = (
            flags["serious_or_life_threatening_condition"]
            or flags["serious_endpoint"]
            or flags["vulnerable_age_population"]
        )
        # Tier 2 — late phase requires at least one additional risk feature
        tier2 = flags["late_phase"] and (
            flags["large_enrollment"]
            or flags["long_duration"]
            or flags["serious_or_life_threatening_condition"]
            or flags["serious_endpoint"]
            or flags["regulated_intervention"]
        )
        # Tier 3 — regulated intervention requires scale (large or long)
        tier3 = flags["regulated_intervention"] and (
            flags["large_enrollment"]
            or flags["long_duration"]
        )

        if is_interventional and tier1:
            dmc_expected, dmc_tier = True, "tier1"
        elif is_interventional and tier2:
            dmc_expected, dmc_tier = True, "tier2"
        elif is_interventional and tier3:
            dmc_expected, dmc_tier = True, "tier3"
        else:
            dmc_expected, dmc_tier = False, ""

        # Strict: standalone Tier 1 indicator AND at least one compound criterion.
        # Excludes trials that qualify only on vulnerable population or late phase
        # without the corroborating compound evidence.
        dmc_expected_strict = is_interventional and tier1 and (tier2 or tier3)

        # ── Rule description ────────────────────────────────────────────────────
        if not is_interventional:
            if reasons:
                rule = "Non-interventional study; guidance-feature flags retained but DMC indicated is false."
            else:
                rule = "Non-interventional study without guidance features."
        elif dmc_expected:
            tier_descriptions = {
                "tier1": (
                    "Regulatory-risk proxy (Tier 1 — standalone): serious/life-threatening "
                    "condition, serious endpoint, or vulnerable population."
                ),
                "tier2": (
                    "Regulatory-risk proxy (Tier 2 — compound): late-phase trial with at "
                    "least one additional risk feature."
                ),
                "tier3": (
                    "Regulatory-risk proxy (Tier 3 — compound): regulated intervention with "
                    "large enrollment or long duration."
                ),
            }
            rule = tier_descriptions[dmc_tier]
        elif dmc_expected_broad:
            rule = (
                "Interventional trial: below strict proxy threshold; qualifies only under "
                "broad single-feature rule (sensitivity analysis)."
            )
        else:
            rule = "Interventional trial without sufficient regulatory-risk features."

        active_reasons = reasons if dmc_expected else []
        return DMCExpectedResult(
            dmc_expected=dmc_expected,
            dmc_expected_reason_count=len(active_reasons),
            dmc_expected_reasons=active_reasons,
            dmc_expected_reason_labels=[REASON_LABELS[r] for r in active_reasons],
            dmc_expected_rule=rule,
            dmc_expected_tier=dmc_tier,
            dmc_expected_strict=dmc_expected_strict,
            dmc_expected_broad=dmc_expected_broad,
            is_interventional_trial=is_interventional,
            dmc_expected_late_phase=flags["late_phase"],
            dmc_expected_serious_condition=flags["serious_or_life_threatening_condition"],
            dmc_expected_vulnerable_population=flags["vulnerable_age_population"],
            dmc_expected_large_enrollment=flags["large_enrollment"],
            dmc_expected_regulated_intervention=flags["regulated_intervention"],
            dmc_expected_long_duration=flags["long_duration"],
            dmc_expected_serious_endpoint=flags["serious_endpoint"],
            trial_duration_months=round(duration_months, 2) if duration_months is not None else None,
            dmc_expected_llm_used=llm_used,
            dmc_expected_llm_rationale=llm_rationale,
        )

    # ── Vectorised batch classification ─────────────────────────────────────

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all rows using vectorised pandas operations.

        Rule-based flags are computed once across the full DataFrame with
        str.contains and numeric comparisons — no Python loop. The LLM is
        called only for interventional trials where BOTH serious_condition
        AND serious_endpoint are False after the rule pass, minimising API
        round-trips.

        Returns a DataFrame of DMCExpectedResult fields aligned to df.index.
        """
        idx = df.index

        is_interventional = (
            df.get("study_type", pd.Series("", index=idx))
            .fillna("").str.strip().str.lower().eq("interventional")
        )
        late_phase = (
            df.get("phase", pd.Series("", index=idx))
            .fillna("").str.contains(r"\bPhase\s*[34]\b", case=False, regex=True)
        )
        large_enrollment = (
            pd.to_numeric(
                df.get("enrollment", pd.Series(np.nan, index=idx)), errors="coerce"
            ) >= self.large_enrollment_threshold
        ).fillna(False)

        start_s = df.get("start_date", pd.Series(pd.NaT, index=idx))
        end_s = df.get("completion_date", pd.Series(pd.NaT, index=idx)).where(
            df.get("completion_date", pd.Series(pd.NaT, index=idx)).notna(),
            df.get("primary_completion_date", pd.Series(pd.NaT, index=idx)),
        )
        duration_months = self._duration_months_series(start_s, end_s)
        long_duration = (duration_months >= self.long_duration_months_threshold).fillna(False)

        regulated_intervention = (
            df.get("intervention_types", pd.Series("", index=idx))
            .fillna("").str.contains(self._REGULATED_INTERVENTION_PATTERN, na=False)
        )

        cond_text = self._join_text_columns(
            df, ["conditions", "keywords", "therapeutic_area", "brief_title", "official_title"]
        )
        serious_condition = cond_text.str.contains(self._SERIOUS_CONDITION_COMBINED, na=False)

        ep_text = self._join_text_columns(
            df,
            ["primary_outcomes", "primary_outcome_time_frames",
             "secondary_outcomes", "secondary_outcome_time_frames"],
        )
        serious_endpoint = ep_text.str.contains(self._SERIOUS_ENDPOINT_COMBINED, na=False)

        vulnerable_population = self._vulnerable_age_series(df)

        # LLM — parallel calls with content-hash cache to skip duplicate text.
        # Compute preliminary tier logic first so the LLM is restricted to trials
        # that don't already qualify under any rule: adding a serious flag can only
        # change the classification for currently-false dmc_expected rows.
        _prelim_tier1 = serious_condition | serious_endpoint | vulnerable_population
        _prelim_tier2 = late_phase & (
            large_enrollment | long_duration | serious_condition
            | serious_endpoint | regulated_intervention
        )
        _prelim_tier3 = regulated_intervention & (large_enrollment | long_duration)
        _prelim_dmc_expected = is_interventional & (_prelim_tier1 | _prelim_tier2 | _prelim_tier3)

        llm_used = pd.Series(False, index=idx)
        llm_rationale = pd.Series("", index=idx)
        if self.llm_client:
            needs_llm = (
                is_interventional
                & ~_prelim_dmc_expected
                & ~serious_condition
                & ~serious_endpoint
            )
            llm_indices = idx[needs_llm].tolist()

            _LLM_FIELDS = (
                "conditions", "keywords", "therapeutic_area",
                "brief_title", "official_title",
                "primary_outcomes", "secondary_outcomes",
            )
            _cache: Dict[str, Optional[Dict[str, Any]]] = {}

            def _call_one(row_idx: Any) -> tuple:
                row_dict = df.loc[row_idx].to_dict()
                fingerprint = hashlib.md5(
                    "|".join(str(row_dict.get(f, "")) for f in _LLM_FIELDS).encode()
                ).hexdigest()
                if fingerprint in _cache:
                    return row_idx, _cache[fingerprint]
                flags_dict: Dict[str, bool] = {
                    "serious_or_life_threatening_condition": False,
                    "serious_endpoint": False,
                    "late_phase": bool(late_phase.loc[row_idx]),
                    "vulnerable_age_population": bool(vulnerable_population.loc[row_idx]),
                    "large_enrollment": bool(large_enrollment.loc[row_idx]),
                    "regulated_intervention": bool(regulated_intervention.loc[row_idx]),
                    "long_duration": bool(long_duration.loc[row_idx]),
                }
                result = self._classify_with_llm(row_dict, flags_dict)
                _cache[fingerprint] = result
                return row_idx, result

            n_workers = min(self.LLM_MAX_WORKERS, max(1, len(llm_indices)))
            llm_updates: Dict[Any, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for fut in as_completed(pool.submit(_call_one, i) for i in llm_indices):
                    row_idx, res = fut.result()
                    if res:
                        llm_updates[row_idx] = res

            for row_idx, res in llm_updates.items():
                if res["flags"].get("serious_or_life_threatening_condition"):
                    serious_condition.at[row_idx] = True
                if res["flags"].get("serious_endpoint"):
                    serious_endpoint.at[row_idx] = True
                llm_used.at[row_idx] = True
                llm_rationale.at[row_idx] = res.get("rationale", "")

        # Tier logic (vectorised)
        tier1 = serious_condition | serious_endpoint | vulnerable_population
        tier2 = late_phase & (
            large_enrollment | long_duration | serious_condition
            | serious_endpoint | regulated_intervention
        )
        tier3 = regulated_intervention & (large_enrollment | long_duration)

        dmc_expected = is_interventional & (tier1 | tier2 | tier3)
        dmc_expected_strict = is_interventional & tier1 & (tier2 | tier3)
        dmc_expected_broad = is_interventional & (
            late_phase | serious_condition | vulnerable_population
            | large_enrollment | regulated_intervention | long_duration | serious_endpoint
        )

        dmc_tier = pd.Series("", index=idx, dtype=str)
        dmc_tier = dmc_tier.where(~(is_interventional & tier1), "tier1")
        dmc_tier = dmc_tier.where(~(is_interventional & ~tier1 & tier2), "tier2")
        dmc_tier = dmc_tier.where(~(is_interventional & ~tier1 & ~tier2 & tier3), "tier3")
        dmc_tier = dmc_tier.where(dmc_expected, "")

        # ── Reason lists (vectorised) ────────────────────────────────────────
        _flags_df = pd.DataFrame({
            "late_phase": late_phase,
            "serious_or_life_threatening_condition": serious_condition,
            "vulnerable_age_population": vulnerable_population,
            "large_enrollment": large_enrollment,
            "regulated_intervention": regulated_intervention,
            "long_duration": long_duration,
            "serious_endpoint": serious_endpoint,
        }, index=idx)
        # Zero out flags for non-indicated rows so reasons list stays empty
        _mask = pd.DataFrame(
            np.repeat(dmc_expected.to_numpy()[:, None], len(DMC_EXPECTED_REASONS), axis=1),
            index=idx, columns=list(DMC_EXPECTED_REASONS),
        )
        _active = _flags_df.where(_mask, False)

        # Bitmask lookup — precompute all 2^7 = 128 combinations once,
        # then map each row's flag pattern to a precomputed string (zero Python loops).
        _reasons_list = list(DMC_EXPECTED_REASONS)
        _reason_lut: Dict[int, str] = {}
        _label_lut:  Dict[int, str] = {}
        for bits in range(1 << len(_reasons_list)):
            active_r = [r for i, r in enumerate(_reasons_list) if bits & (1 << i)]
            _reason_lut[bits] = "; ".join(active_r)
            _label_lut[bits]  = "; ".join(REASON_LABELS[r] for r in active_r)

        _powers = np.array([1 << i for i in range(len(_reasons_list))], dtype=np.int32)
        bitmask = (_active.to_numpy(dtype=np.int32) * _powers).sum(axis=1)

        reason_list_col = pd.Series(bitmask, index=idx).map(_reason_lut)
        label_list_col  = pd.Series(bitmask, index=idx).map(_label_lut)
        count_col       = _active.sum(axis=1).astype(int)

        # ── Rule descriptions (vectorised with np.select) ────────────────────
        _T1 = "Regulatory-risk proxy (Tier 1 — standalone): serious/life-threatening condition, serious endpoint, or vulnerable population."
        _T2 = "Regulatory-risk proxy (Tier 2 — compound): late-phase trial with at least one additional risk feature."
        _T3 = "Regulatory-risk proxy (Tier 3 — compound): regulated intervention with large enrollment or long duration."
        _BROAD = "Interventional trial: below primary proxy threshold; qualifies only under broad single-feature rule."
        _NO_RISK = "Interventional trial without sufficient regulatory-risk features."
        _NON_INT = "Non-interventional study without guidance features."

        rule_col = pd.Series(
            np.select(
                [
                    ~is_interventional,
                    dmc_expected & dmc_tier.eq("tier1"),
                    dmc_expected & dmc_tier.eq("tier2"),
                    dmc_expected & dmc_tier.eq("tier3"),
                    ~dmc_expected & dmc_expected_broad,
                ],
                [_NON_INT, _T1, _T2, _T3, _BROAD],
                default=_NO_RISK,
            ),
            index=idx,
        )

        return pd.DataFrame({
            "dmc_expected": dmc_expected,
            "dmc_expected_reason_count": count_col,
            "dmc_expected_reasons": reason_list_col,
            "dmc_expected_reason_labels": label_list_col,
            "dmc_expected_rule": rule_col,
            "dmc_expected_tier": dmc_tier,
            "dmc_expected_strict": dmc_expected_strict,
            "dmc_expected_broad": dmc_expected_broad,
            "is_interventional_trial": is_interventional,
            "dmc_expected_late_phase": late_phase,
            "dmc_expected_serious_condition": serious_condition,
            "dmc_expected_vulnerable_population": vulnerable_population,
            "dmc_expected_large_enrollment": large_enrollment,
            "dmc_expected_regulated_intervention": regulated_intervention,
            "dmc_expected_long_duration": long_duration,
            "dmc_expected_serious_endpoint": serious_endpoint,
            "trial_duration_months": duration_months.round(2),
            "dmc_expected_llm_used": llm_used,
            "dmc_expected_llm_rationale": llm_rationale,
        }, index=idx)

    @staticmethod
    def _join_text_columns(df: pd.DataFrame, cols: List[str]) -> pd.Series:
        """Join multiple text columns into a single string per row (vectorised)."""
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.Series("", index=df.index)
        parts = [df[c].fillna("").astype(str) for c in existing]
        result = parts[0]
        for s in parts[1:]:
            result = result + " " + s
        return result

    @staticmethod
    def _duration_months_series(start: pd.Series, end: pd.Series) -> pd.Series:
        """Vectorised month duration. Returns NaN when dates are missing or reversed."""
        start_dt = pd.to_datetime(start, errors="coerce")
        end_dt = pd.to_datetime(end, errors="coerce")
        diff_days = (end_dt - start_dt).dt.days
        months = diff_days / 30.4375
        return months.where(
            end_dt.notna() & start_dt.notna() & (end_dt >= start_dt), np.nan
        )

    def _vulnerable_age_series(self, df: pd.DataFrame) -> pd.Series:
        """Fully vectorised vulnerable age flag — no apply()."""
        min_age = self._age_to_years_series(
            df.get("minimum_age", pd.Series(dtype=object, index=df.index))
        )
        max_age = self._age_to_years_series(
            df.get("maximum_age", pd.Series(dtype=object, index=df.index))
        )
        includes_children = (min_age < 18).fillna(False) | (
            min_age.isna() & (max_age < 18).fillna(False)
        )
        includes_elderly = (min_age >= self.elderly_age_threshold).fillna(False)
        return (includes_children | includes_elderly).fillna(False)

    @staticmethod
    def _age_to_years_series(series: pd.Series) -> pd.Series:
        """Vectorised age-string → years (mirrors _age_to_years but operates on a Series)."""
        s = series.fillna("").astype(str).str.strip()
        missing = s.str.lower().isin(
            {"", "nan", "none", "null", "<na>", "n/a", "na", "no maximum"}
        )
        m = s.str.extract(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)?", expand=True)
        number = pd.to_numeric(m[0], errors="coerce")
        unit = m[1].fillna("years").str.lower()

        years = np.select(
            [
                unit.str.startswith("year"),
                unit.str.startswith("month"),
                unit.str.startswith("week"),
                unit.str.startswith("day"),
                unit.str.startswith("hour"),
                unit.str.startswith("minute"),
            ],
            [
                number,
                number / 12.0,
                number / 52.1775,
                number / 365.25,
                number / (365.25 * 24.0),
                number / (365.25 * 24.0 * 60.0),
            ],
            default=number,  # no unit → assume years
        )
        result = pd.Series(years, index=series.index, dtype=float)
        return result.where(~missing, np.nan)

    # ── Per-row LLM helper ───────────────────────────────────────────────────

    def _classify_with_llm(
        self,
        row: Mapping[str, Any],
        flags: Dict[str, bool],
    ) -> Optional[Dict[str, Any]]:
        if not self.llm_client:
            return None

        messages = self.build_llm_messages(row, flags)
        try:
            raw = self.llm_client.classify(messages)
        except Exception:
            return None

        # Validate with Pydantic when available; fall back to manual parsing
        if _HAS_PYDANTIC:
            try:
                parsed = DMCLLMResponse.model_validate(raw)
                confidence = parsed.confidence
                condition_flag = parsed.serious_or_life_threatening_condition
                endpoint_flag = parsed.serious_endpoint
                rationale = parsed.rationale
            except Exception:
                return None
        else:
            confidence = self._clamp_confidence(raw.get("confidence", 0.0))
            condition_flag = self._bool_from_value(
                raw.get("serious_or_life_threatening_condition", raw.get("serious_condition", False))
            )
            endpoint_flag = self._bool_from_value(raw.get("serious_endpoint", False))
            rationale = str(raw.get("rationale", "")).strip()

        if confidence < self.llm_confidence_threshold:
            return None
        if not condition_flag and not endpoint_flag:
            return None

        return {
            "flags": {
                "serious_or_life_threatening_condition": (
                    flags["serious_or_life_threatening_condition"] or condition_flag
                ),
                "serious_endpoint": flags["serious_endpoint"] or endpoint_flag,
            },
            "rationale": rationale,
        }

    @classmethod
    def build_llm_messages(
        cls,
        row: Mapping[str, Any],
        flags: Dict[str, bool],
    ) -> List[Mapping[str, str]]:
        context = {
            "conditions": " ".join(cls._as_text_list(row.get("conditions"))),
            "keywords": " ".join(cls._as_text_list(row.get("keywords"))),
            "therapeutic_area": str(row.get("therapeutic_area", "")),
            "brief_title": str(row.get("brief_title", "")),
            "official_title": str(row.get("official_title", "")),
            "primary_outcomes": " ".join(cls._as_text_list(row.get("primary_outcomes"))),
            "primary_outcome_time_frames": " ".join(cls._as_text_list(row.get("primary_outcome_time_frames"))),
            "secondary_outcomes": " ".join(cls._as_text_list(row.get("secondary_outcomes"))),
            "secondary_outcome_time_frames": " ".join(cls._as_text_list(row.get("secondary_outcome_time_frames"))),
        }

        system = (
            "You read ClinicalTrials.gov trial metadata and decide whether the trial "
            "involves a serious or life-threatening condition and whether it measures "
            "a serious clinical endpoint. Use only the provided metadata. Return JSON only."
        )
        user_payload = {
            "serious_condition_text": context["conditions"],
            "endpoint_text": " ".join(
                [
                    context["primary_outcomes"],
                    context["primary_outcome_time_frames"],
                    context["secondary_outcomes"],
                    context["secondary_outcome_time_frames"],
                ]
            ).strip(),
            "trial_context": {
                "brief_title": context["brief_title"],
                "official_title": context["official_title"],
                "keywords": context["keywords"],
                "therapeutic_area": context["therapeutic_area"],
            },
            "rule_flags": {
                "serious_or_life_threatening_condition": flags["serious_or_life_threatening_condition"],
                "serious_endpoint": flags["serious_endpoint"],
            },
            "instructions": [
                "Return only JSON with the required fields.",
                "Do not add extra narrative outside the JSON object.",
                "Set serious_or_life_threatening_condition to true only if the trial clearly involves a serious or life-threatening condition.",
                "Set serious_endpoint to true only if the trial clearly measures a serious clinical endpoint.",
                "Set confidence between 0 and 1.",
                "Keep rationale under 20 words.",
            ],
            "required_json_shape": {
                "serious_or_life_threatening_condition": "boolean",
                "serious_endpoint": "boolean",
                "confidence": "number from 0 to 1",
                "rationale": "brief explanation",
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ]

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, confidence))

    @staticmethod
    def _bool_from_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    @staticmethod
    def _is_interventional(value: Any) -> bool:
        return str(value or "").strip().lower() == "interventional"

    @staticmethod
    def _is_late_phase(value: Any) -> bool:
        text = str(value or "")
        return bool(re.search(r"\bPhase 3\b", text, re.IGNORECASE)) or bool(
            re.search(r"\bPhase 4\b", text, re.IGNORECASE)
        )

    def _has_serious_condition(self, row: Mapping[str, Any]) -> bool:
        values = []
        for key in ("conditions", "keywords", "therapeutic_area", "brief_title", "official_title"):
            values.extend(self._as_text_list(row.get(key)))
        text = " ".join(values)
        return any(pattern.search(text) for pattern in self.SERIOUS_CONDITION_PATTERNS)

    def _has_vulnerable_age(self, row: Mapping[str, Any]) -> bool:
        minimum_age = self._age_to_years(row.get("minimum_age"))
        maximum_age = self._age_to_years(row.get("maximum_age"))

        includes_children = False
        if minimum_age is not None and minimum_age < 18.0:
            includes_children = True
        if minimum_age is None and maximum_age is not None and maximum_age < 18.0:
            includes_children = True

        # Only flag as exclusively elderly when the trial REQUIRES participants
        # to be >= threshold; maximum_age >= 65 is nearly universal and too broad.
        includes_exclusively_elderly = (
            minimum_age is not None and minimum_age >= self.elderly_age_threshold
        )

        return includes_children or includes_exclusively_elderly

    def _has_large_enrollment(self, value: Any) -> bool:
        enrollment = self._to_float(value)
        return enrollment is not None and enrollment >= self.large_enrollment_threshold

    def _has_regulated_intervention(self, value: Any) -> bool:
        intervention_types = {item.upper().strip() for item in self._as_text_list(value)}
        return bool(intervention_types & self.REGULATED_INTERVENTION_TYPES)

    def _has_serious_endpoint(self, row: Mapping[str, Any]) -> bool:
        values = []
        for key in (
            "primary_outcomes",
            "primary_outcome_time_frames",
            "secondary_outcomes",
            "secondary_outcome_time_frames",
        ):
            values.extend(self._as_text_list(row.get(key)))
        text = " ".join(values)
        return any(pattern.search(text) for pattern in self.SERIOUS_ENDPOINT_PATTERNS)

    @classmethod
    def _as_text_list(cls, value: Any) -> List[str]:
        if cls._is_missing(value):
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (SyntaxError, ValueError):
                    parsed = stripped
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item).strip() for item in parsed if not cls._is_missing(item)]
            return [stripped]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if not cls._is_missing(item)]
        return [str(value).strip()]

    @classmethod
    def _age_to_years(cls, value: Any) -> Optional[float]:
        if cls._is_missing(value):
            return None
        text = str(value).strip()
        if text.lower() in {"n/a", "na", "no maximum", "none"}:
            return None

        match = re.search(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)?", text)
        if not match:
            return None

        number = float(match.group(1))
        unit = (match.group(2) or "years").lower()
        if unit.startswith("year"):
            return number
        if unit.startswith("month"):
            return number / 12.0
        if unit.startswith("week"):
            return number / 52.1775
        if unit.startswith("day"):
            return number / 365.25
        if unit.startswith("hour"):
            return number / (365.25 * 24.0)
        if unit.startswith("minute"):
            return number / (365.25 * 24.0 * 60.0)
        return None

    @classmethod
    def _duration_months(cls, start_value: Any, end_value: Any) -> Optional[float]:
        start = cls._parse_partial_date(start_value)
        end = cls._parse_partial_date(end_value)
        if not start or not end or end < start:
            return None
        return (end - start).days / 30.4375

    @classmethod
    def _parse_partial_date(cls, value: Any) -> Optional[datetime]:
        if cls._is_missing(value):
            return None
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @classmethod
    def _to_float(cls, value: Any) -> Optional[float]:
        if cls._is_missing(value):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number):
            return None
        return number

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        text = str(value).strip().lower()
        return text in {"", "nan", "none", "null", "<na>"}
