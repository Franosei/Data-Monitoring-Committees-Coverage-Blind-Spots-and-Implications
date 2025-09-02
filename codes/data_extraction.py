"""
ClinicalTrials.gov v2 Downloader — DMC Coverage Dataset

- Correct v2 pagination: read `x-next-page-token` from response headers,
  send it back as `pageToken` in the next request.
- Pulls only trials whose overall status is one of:
  COMPLETED, TERMINATED, WITHDRAWN, SUSPENDED (server-filtered where possible).
- Filters by start date in [2010-01-01, 2025-12-31] (inclusive; tolerant of YYYY-MM / YYYY).
- Extracts fields needed for DMC oversight analysis.
- De-duplicates NCT IDs across pages and across runs (optional state file).
- Robust backoff & retry; optional page cap for testing.
"""

from __future__ import annotations
import os
import time
import json
import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple
from datetime import datetime

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("CTGV2DMCDownloader")


class CTGV2DMCDownloader:
    """
    Downloader for ClinicalTrials.gov v2 studies tailored for DMC coverage analysis.

    Parameters
    ----------
    storage_seen_path : Optional[str]
        Path to a newline-delimited file of NCT IDs to avoid re-downloading across runs.
    page_size : int
        Page size for the v2 API (`max` is 1000). Defaults to 1000.
    request_timeout : int
        Seconds before timing out a single HTTP request. Defaults to 60.
    backoff_seconds : float
        Initial backoff (exponential) between retries on 429/5xx. Defaults to 1.0.
    enable_server_filters : bool
        If True, sends `filter.overallStatus` to the API. If the server rejects this
        (HTTP 400 unknown parameter), the client falls back to client-side filtering.
    base_url : str
        v2 `studies` endpoint.
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    # Fields needed for this analysis
    FIELDS: Tuple[str, ...] = (
        "protocolSection.identificationModule.nctId",
        "protocolSection.oversightModule.oversightHasDmc",
        "protocolSection.designModule.studyType",
        "protocolSection.designModule.phases",
        "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
        "protocolSection.statusModule.overallStatus",
        "protocolSection.statusModule.whyStopped",
        "protocolSection.statusModule.startDateStruct.date",
        "protocolSection.statusModule.primaryCompletionDateStruct.date",
        "protocolSection.conditionsModule.conditions",
    )

    # Default statuses for “completed/terminated/withdrawn/suspended only”
    DEFAULT_ELIGIBLE_STATUSES = ("COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED")

    def __init__(
        self,
        storage_seen_path: Optional[str] = "seen_ids.txt",
        page_size: int = 1000,
        request_timeout: int = 60,
        backoff_seconds: float = 1.0,
        enable_server_filters: bool = True,
        base_url: str = BASE_URL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.page_size = int(page_size)
        self.request_timeout = int(request_timeout)
        self.backoff_seconds = float(backoff_seconds)
        self.enable_server_filters = bool(enable_server_filters)

        self.storage_seen_path = storage_seen_path
        self.seen_ids: Set[str] = set()
        if storage_seen_path:
            self._load_seen_ids(storage_seen_path)

    # Public API

    def fetch_dataframe(
        self,
        start_date_from: str = "2010-01-01",
        start_date_to: str = "2025-12-31",
        statuses: Iterable[str] = DEFAULT_ELIGIBLE_STATUSES,
        max_pages: Optional[int] = None,
        query_term: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve studies for the DMC dataset.

        Parameters
        ----------
        start_date_from : str
            Inclusive lower bound for `statusModule.startDateStruct.date`.
        start_date_to : str
            Inclusive upper bound for `statusModule.startDateStruct.date`.
        statuses : Iterable[str]
            Only keep studies whose `overallStatus` is in this set (case-insensitive).
        max_pages : Optional[int]
            If provided, stop after this many pages (useful for testing).
        query_term : Optional[str]
            Optional Essie/legacy search expression to reduce result set (sent as `query.term`).

        Returns
        -------
        pandas.DataFrame
            One row per unique NCT ID with these columns:
            nct_id, has_dmc, phase, study_type, sponsor_class, overall_status,
            why_stopped, start_date, start_year, primary_completion_date,
            conditions (list[str]), therapeutic_area
        """
        statuses_upper = tuple(s.upper() for s in statuses)
        logger.info("Starting fetch %s → %s; statuses=%s",
                    start_date_from, start_date_to, ",".join(statuses_upper))

        rows: List[Dict] = []
        page_token: Optional[str] = None
        page_counter = 0
        use_server_filter = self.enable_server_filters

        while True:
            if max_pages is not None and page_counter >= max_pages:
                logger.info("Reached max_pages=%d. Stopping.", max_pages)
                break

            try:
                data, next_token = self._get_page(
                    page_token=page_token,
                    query_term=query_term,
                    use_server_filter=use_server_filter,
                    statuses=statuses_upper,
                )
            except ValueError as e:
                # Fallback: if server filter caused 400, retry without it
                if "SERVER_FILTER_UNSUPPORTED" in str(e) and use_server_filter:
                    logger.warning("Server-side status filter not supported; retrying without it.")
                    use_server_filter = False
                    continue
                raise

            page_counter += 1
            if not data:
                logger.info("No data on this page. Stopping.")
                break

            for study in data:
                rec = self._extract_record(study)
                if rec is None:
                    continue

                # client-side filters
                if not self._within_start_window(rec.get("start_date"), start_date_from, start_date_to):
                    continue

                if not rec.get("overall_status"):
                    continue
                if rec["overall_status"].upper() not in statuses_upper:
                    continue

                nct_id = rec["nct_id"]
                if nct_id in self.seen_ids:
                    continue

                self.seen_ids.add(nct_id)
                rows.append(rec)

            logger.info("Page %d: accumulated %d unique trials.", page_counter, len(rows))

            if not next_token:
                logger.info("No next page token. Finished.")
                break

            page_token = next_token  # IMPORTANT: v2 expects `pageToken` param in next request

        self.save_seen_ids()

        df = pd.DataFrame.from_records(rows)
        if df.empty:
            logger.info("Final dataset size: 0")
            return df

        df = df.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)
        df["has_dmc"] = df["has_dmc"].astype("boolean")
        # start_year for year-on-year trend checks
        df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year

        logger.info("Final dataset size: %d unique trials.", len(df))
        return df

    def save_seen_ids(self) -> None:
        """Persist seen NCT IDs to disk if `storage_seen_path` was provided."""
        if not self.storage_seen_path:
            return
        try:
            with open(self.storage_seen_path, "w", encoding="utf-8") as f:
                for nct in sorted(self.seen_ids):
                    f.write(nct + "\n")
        except Exception as exc:
            logger.warning("Failed to save seen IDs to %s: %s", self.storage_seen_path, exc)

    def reset_seen_ids(self) -> None:
        """Clear in-memory and on-disk seen-ID set."""
        self.seen_ids.clear()
        if self.storage_seen_path and os.path.exists(self.storage_seen_path):
            try:
                os.remove(self.storage_seen_path)
            except Exception as exc:
                logger.warning("Failed to remove %s: %s", self.storage_seen_path, exc)

    # Internals

    def _get_page(
        self,
        page_token: Optional[str],
        query_term: Optional[str],
        use_server_filter: bool,
        statuses: Iterable[str],
        max_retries: int = 6,
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Request a single page of studies with selected fields.

        Pagination contract (v2):
          - Response header `x-next-page-token` holds the cursor for the next page.
          - Send it back as `pageToken` in the next request.

        If `use_server_filter` is True, attempts `filter.overallStatus`.
        If the server rejects the parameter with HTTP 400, raises ValueError("SERVER_FILTER_UNSUPPORTED").
        """
        params = {
            "fields": ",".join(self.FIELDS),
            "pageSize": self.page_size,
            "countTotal": "false",  # set "true" if you want approximate total count
        }
        if page_token:
            # IMPORTANT: correct parameter name for v2 is `pageToken`
            params["pageToken"] = page_token
        if query_term:
            params["query.term"] = query_term
        if use_server_filter and statuses:
            # Try server-side status filter; fallback is handled by caller on 400
            params["filter.overallStatus"] = ",".join(statuses)

        backoff = self.backoff_seconds

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(self.base_url, params=params, timeout=self.request_timeout)
            except requests.RequestException as e:
                logger.warning("Request error (attempt %d/%d): %s", attempt, max_retries, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = float(resp.headers.get("Retry-After", backoff))
                logger.warning(
                    "HTTP %s (attempt %d/%d). Retrying in %.1fs.",
                    resp.status_code, attempt, max_retries, retry_after
                )
                time.sleep(retry_after)
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code == 400:
                text = resp.text[:300]
                if "unknown parameter" in text.lower() and "filter.overallstatus" in text.lower():
                    # Signal to caller to retry without server filter
                    raise ValueError("SERVER_FILTER_UNSUPPORTED")
                logger.error("HTTP 400: %s", text)
                return [], None

            if resp.status_code != 200:
                logger.error("HTTP %s: %s", resp.status_code, resp.text[:250])
                return [], None

            try:
                payload = resp.json()
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON.")
                return [], None

            studies = payload.get("studies", [])
            # Correct header name for pagination cursor
            next_token = resp.headers.get("x-next-page-token") or payload.get("nextPageToken") or None
            return studies, next_token

        logger.error("Exceeded max retries without success.")
        return [], None

    def _extract_record(self, study: Dict) -> Optional[Dict]:
        """
        Extract a tidy record from a raw study dict; return None if missing NCT ID.
        """
        g = study.get("protocolSection", {})
        nct_id = self._get(g, "identificationModule.nctId")
        if not nct_id:
            return None

        has_dmc = self._get(g, "oversightModule.oversightHasDmc")
        study_type = self._get(g, "designModule.studyType")
        phase = self._phase_join(self._get(g, "designModule.phases"))
        sponsor_class = self._get(g, "sponsorCollaboratorsModule.leadSponsor.class")
        overall_status = self._get(g, "statusModule.overallStatus")
        why_stopped = self._get(g, "statusModule.whyStopped")
        start_date = self._get(g, "statusModule.startDateStruct.date")
        primary_completion = self._get(g, "statusModule.primaryCompletionDateStruct.date")

        conditions = self._get(g, "conditionsModule.conditions") or []
        if not isinstance(conditions, list):
            conditions = [conditions] if conditions else []
        therapeutic_area = self._classify_therapeutic_area(conditions)

        return {
            "nct_id": nct_id,
            "has_dmc": has_dmc,
            "phase": phase,
            "study_type": study_type,
            "sponsor_class": sponsor_class,
            "overall_status": overall_status,
            "why_stopped": why_stopped,
            "start_date": start_date,
            "primary_completion_date": primary_completion,
            "conditions": conditions,
            "therapeutic_area": therapeutic_area,
        }

    @staticmethod
    def _get(d: Dict, dotted: str):
        """Safe nested get using 'a.b.c' dotted keys."""
        cur = d
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    @staticmethod
    def _phase_join(phases):
        """
        Normalize phases to a single string; handles lists like ["PHASE1", "PHASE2"]
        -> "Phase 1/2". Returns None if not available.
        """
        if phases is None:
            return None
        if isinstance(phases, list):
            pretty = [CTGV2DMCDownloader._phase_pretty(p) for p in phases]
            pretty = [x for x in pretty if x]
            return "/".join(sorted(set(pretty))) if pretty else None
        return CTGV2DMCDownloader._phase_pretty(phases)

    @staticmethod
    def _phase_pretty(p: str) -> Optional[str]:
        if not p:
            return None
        p = str(p).upper()
        mapping = {
            "EARLY_PHASE1": "Early Phase 1",
            "PHASE1": "Phase 1",
            "PHASE2": "Phase 2",
            "PHASE3": "Phase 3",
            "PHASE4": "Phase 4",
            "NA": "N/A",
        }
        return mapping.get(p, p.title())

    @staticmethod
    def _within_start_window(start_date_iso: Optional[str], lo_str: str, hi_str: str) -> bool:
        """
        True if start_date_iso ∈ [lo, hi], with tolerant parsing of YYYY-MM and YYYY.
        Missing start dates return False to keep time series clean.
        """
        if not start_date_iso:
            return False

        def parse_any(s: str) -> Optional[datetime]:
            for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            return None

        x = parse_any(start_date_iso)
        if not x:
            return False

        lo = datetime.strptime(lo_str, "%Y-%m-%d")
        hi = datetime.strptime(hi_str, "%Y-%m-%d")
        return lo <= x <= hi

    @staticmethod
    def _classify_therapeutic_area(conditions: Iterable[str]) -> str:
        """
        Lightweight therapeutic area classifier from condition strings.
        Extend/adjust rules as needed for your portfolio.
        """
        text = " ".join([c or "" for c in conditions]).lower()

        rules = [
            ("Oncology", ["cancer", "carcinoma", "lymphoma", "leukemia", "tumor", "melanoma", "neoplasm"]),
            ("Cardiovascular", ["cardio", "heart", "hypertension", "myocard", "atrial", "stroke", "vascular"]),
            ("Neurology", ["neuro", "alzheimer", "parkinson", "epilep", "multiple sclerosis", "ms ", "migraine"]),
            ("Psychiatry", ["depress", "schizo", "bipolar", "anxiety", "adhd", "autism", "ptsd"]),
            ("Endocrine/Metabolic", ["diabet", "metabolic", "thyroid", "obesity", "hyperlip", "dyslipid"]),
            ("Infectious Disease", ["covid", "influenza", "hiv", "hepatitis", "malaria", "tubercu", "infection"]),
            ("Respiratory", ["asthma", "copd", "bronch", "respiratory", "pulmon"]),
            ("Gastroenterology", ["crohn", "ulcerative colitis", "ibs", "liver", "hepat", "gi "]),
            ("Rheumatology/Immunology", ["rheumat", "lupus", "psoriasis", "autoimmun", "arthritis"]),
            ("Dermatology", ["dermat", "eczema", "atopic", "acne", "skin"]),
            ("Obstetrics/Gynecology", ["pregnan", "obstet", "gyneco", "endometri", "fertility", "ivf"]),
            ("Ophthalmology", ["retina", "glaucoma", "ophthal", "macular", "uveitis"]),
            ("Hematology", ["hemato", "anemia", "hemoph", "sickle", "thrombo", "coagul"]),
            ("Genetic/Rare", ["rare", "orphan", "duchenne", "huntington", "lysosomal", "sma ", "tay-sachs"]),
        ]

        for label, keys in rules:
            if any(k in text for k in keys):
                return label
        return "Other/Unclassified"

    def _load_seen_ids(self, path: str) -> None:
        """Load seen NCT IDs from a newline-delimited file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    nct = line.strip()
                    if nct:
                        self.seen_ids.add(nct)
            logger.info("Loaded %d previously seen NCT IDs from %s.", len(self.seen_ids), path)
        except FileNotFoundError:
            logger.info("No existing seen-ID file at %s (starting fresh).", path)
        except Exception as exc:
            logger.warning("Failed to load seen IDs from %s: %s", path, exc)



if __name__ == "__main__":
    # Adjust the path if you want a different location for the dedupe cache.
    dl = CTGV2DMCDownloader(storage_seen_path="seen_ids.txt")

    df = dl.fetch_dataframe(
        start_date_from="2010-01-01",
        start_date_to="2025-12-31",
        statuses=("COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"),
        max_pages=None,          # set an int to test first (e.g., 3), then None for full crawl
        query_term=None          # optional Essie expression to narrow; leave None for broad
    )

    out_csv = "data/ctg_dmc_2010_2025_completed_terminated_withdrawn_suspended.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} unique trials to {out_csv}")
