from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


class MatchReasonCode(str, Enum):
    """Canonical reason codes for SEC-CCM linking and alignment outcomes."""

    OK = "OK"
    BAD_INPUT = "BAD_INPUT"
    CIK_NOT_IN_LINK_UNIVERSE = "CIK_NOT_IN_LINK_UNIVERSE"
    AMBIGUOUS_LINK = "AMBIGUOUS_LINK"
    NO_CCM_ROW_FOR_DATE = "NO_CCM_ROW_FOR_DATE"
    OUT_OF_CCM_COVERAGE = "OUT_OF_CCM_COVERAGE"


@dataclass(frozen=True)
class SecCcmJoinSpecV1:
    """Versioned join-spec persisted with SEC-CCM pre-merge artifacts."""

    version: str = "v1"
    alignment_policy: Literal["NEXT_TRADING_DAY_STRICT", "FIRST_CLOSE_AFTER_ACCEPTANCE"] = (
        "NEXT_TRADING_DAY_STRICT"
    )
    timezone: str = "America/New_York"
    daily_join_enabled: bool = True
    daily_join_source: Literal["CRSP_DAILY", "MERGED_DAILY_PANEL"] = "CRSP_DAILY"
    daily_permno_col: str = "KYPERMNO"
    daily_date_col: str = "CALDT"
    daily_feature_columns: tuple[str, ...] = (
        "RET",
        "RETX",
        "PRC",
        "BIDLO",
        "ASKHI",
        "SHRCD",
        "EXCHCD",
        "VOL",
        "TCAP",
    )
    required_daily_non_null_features: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out_path
