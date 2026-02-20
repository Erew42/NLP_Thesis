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


class PhaseBAlignmentMode(str, Enum):
    """Phase B doc-date alignment policies."""

    NEXT_TRADING_DAY_STRICT = "NEXT_TRADING_DAY_STRICT"
    FILING_DATE_EXACT_OR_NEXT_TRADING = "FILING_DATE_EXACT_OR_NEXT_TRADING"
    FILING_DATE_EXACT_ONLY = "FILING_DATE_EXACT_ONLY"


class PhaseBDailyJoinMode(str, Enum):
    """Phase B daily join policies after aligned_caldt is computed."""

    ASOF_FORWARD = "ASOF_FORWARD"
    EXACT_ON_ALIGNED_DATE = "EXACT_ON_ALIGNED_DATE"


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


@dataclass(frozen=True)
class SecCcmJoinSpecV2:
    """Versioned join-spec with explicit Phase B alignment and daily join modes."""

    version: str = "v2"
    phase_b_alignment_mode: PhaseBAlignmentMode = PhaseBAlignmentMode.NEXT_TRADING_DAY_STRICT
    phase_b_daily_join_mode: PhaseBDailyJoinMode = PhaseBDailyJoinMode.ASOF_FORWARD
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
    daily_join_max_forward_lag_days: int | None = 14

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["phase_b_alignment_mode"] = self.phase_b_alignment_mode.value
        payload["phase_b_daily_join_mode"] = self.phase_b_daily_join_mode.value
        return payload

    def write_json(self, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out_path


SecCcmJoinSpec = SecCcmJoinSpecV1 | SecCcmJoinSpecV2


def normalize_sec_ccm_join_spec(join_spec: SecCcmJoinSpec) -> SecCcmJoinSpecV2:
    """Normalize V1/V2 join-specs to canonical V2."""
    if isinstance(join_spec, SecCcmJoinSpecV2):
        return join_spec
    if not isinstance(join_spec, SecCcmJoinSpecV1):
        raise TypeError(f"Unsupported join spec type: {type(join_spec)!r}")

    if join_spec.alignment_policy == "FIRST_CLOSE_AFTER_ACCEPTANCE":
        raise ValueError(
            "SecCcmJoinSpecV1(alignment_policy='FIRST_CLOSE_AFTER_ACCEPTANCE') is not supported yet. "
            "Use SecCcmJoinSpecV2 with an implemented phase_b_alignment_mode."
        )

    return SecCcmJoinSpecV2(
        phase_b_alignment_mode=PhaseBAlignmentMode.NEXT_TRADING_DAY_STRICT,
        phase_b_daily_join_mode=PhaseBDailyJoinMode.ASOF_FORWARD,
        timezone=join_spec.timezone,
        daily_join_enabled=join_spec.daily_join_enabled,
        daily_join_source=join_spec.daily_join_source,
        daily_permno_col=join_spec.daily_permno_col,
        daily_date_col=join_spec.daily_date_col,
        daily_feature_columns=join_spec.daily_feature_columns,
        required_daily_non_null_features=join_spec.required_daily_non_null_features,
        daily_join_max_forward_lag_days=14,
    )


def make_sec_ccm_join_spec_preset(
    preset_name: Literal["legacy_default", "lm2011_filing_date", "strict_exact_diagnostic"],
    *,
    base: SecCcmJoinSpecV2 | None = None,
) -> SecCcmJoinSpecV2:
    """Build a canonical V2 join-spec preset."""
    seed = base or SecCcmJoinSpecV2()
    if preset_name == "legacy_default":
        return SecCcmJoinSpecV2(
            phase_b_alignment_mode=PhaseBAlignmentMode.NEXT_TRADING_DAY_STRICT,
            phase_b_daily_join_mode=PhaseBDailyJoinMode.ASOF_FORWARD,
            timezone=seed.timezone,
            daily_join_enabled=seed.daily_join_enabled,
            daily_join_source=seed.daily_join_source,
            daily_permno_col=seed.daily_permno_col,
            daily_date_col=seed.daily_date_col,
            daily_feature_columns=seed.daily_feature_columns,
            required_daily_non_null_features=seed.required_daily_non_null_features,
            daily_join_max_forward_lag_days=seed.daily_join_max_forward_lag_days,
        )
    if preset_name == "lm2011_filing_date":
        return SecCcmJoinSpecV2(
            phase_b_alignment_mode=PhaseBAlignmentMode.FILING_DATE_EXACT_OR_NEXT_TRADING,
            phase_b_daily_join_mode=PhaseBDailyJoinMode.EXACT_ON_ALIGNED_DATE,
            timezone=seed.timezone,
            daily_join_enabled=seed.daily_join_enabled,
            daily_join_source=seed.daily_join_source,
            daily_permno_col=seed.daily_permno_col,
            daily_date_col=seed.daily_date_col,
            daily_feature_columns=seed.daily_feature_columns,
            required_daily_non_null_features=seed.required_daily_non_null_features,
            daily_join_max_forward_lag_days=seed.daily_join_max_forward_lag_days,
        )
    if preset_name == "strict_exact_diagnostic":
        return SecCcmJoinSpecV2(
            phase_b_alignment_mode=PhaseBAlignmentMode.FILING_DATE_EXACT_ONLY,
            phase_b_daily_join_mode=PhaseBDailyJoinMode.EXACT_ON_ALIGNED_DATE,
            timezone=seed.timezone,
            daily_join_enabled=seed.daily_join_enabled,
            daily_join_source=seed.daily_join_source,
            daily_permno_col=seed.daily_permno_col,
            daily_date_col=seed.daily_date_col,
            daily_feature_columns=seed.daily_feature_columns,
            required_daily_non_null_features=seed.required_daily_non_null_features,
            daily_join_max_forward_lag_days=seed.daily_join_max_forward_lag_days,
        )
    raise ValueError(f"Unknown join spec preset: {preset_name}")
