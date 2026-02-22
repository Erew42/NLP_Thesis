from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


class MatchReasonCode(str, Enum):
    """Canonical reason codes for SEC-CCM linking and alignment outcomes.

    Attributes:
        OK: Phase A and Phase B completed successfully.
        BAD_INPUT: Required filing identifiers/dates were invalid or missing.
        CIK_NOT_IN_LINK_UNIVERSE: Filing CIK was not present in the CCM link universe.
        AMBIGUOUS_LINK: Multiple equally ranked link candidates remained after tie handling.
        NO_CCM_ROW_FOR_DATE: Link exists, but no usable daily row was available at aligned date.
        OUT_OF_CCM_COVERAGE: Filing date/aligned date was outside covered daily date range.
    """

    OK = "OK"
    BAD_INPUT = "BAD_INPUT"
    CIK_NOT_IN_LINK_UNIVERSE = "CIK_NOT_IN_LINK_UNIVERSE"
    AMBIGUOUS_LINK = "AMBIGUOUS_LINK"
    NO_CCM_ROW_FOR_DATE = "NO_CCM_ROW_FOR_DATE"
    OUT_OF_CCM_COVERAGE = "OUT_OF_CCM_COVERAGE"


class PhaseBAlignmentMode(str, Enum):
    """Phase B document-date alignment policies.

    Attributes:
        NEXT_TRADING_DAY_STRICT: Align to the first trading day strictly after filing date.
        FILING_DATE_EXACT_OR_NEXT_TRADING: Use filing date if trading day, else next trading day.
        FILING_DATE_EXACT_ONLY: Use filing date only when it is a trading day.
    """

    NEXT_TRADING_DAY_STRICT = "NEXT_TRADING_DAY_STRICT"
    FILING_DATE_EXACT_OR_NEXT_TRADING = "FILING_DATE_EXACT_OR_NEXT_TRADING"
    FILING_DATE_EXACT_ONLY = "FILING_DATE_EXACT_ONLY"


class PhaseBDailyJoinMode(str, Enum):
    """Phase B daily join policies after ``aligned_caldt`` is computed.

    Attributes:
        ASOF_FORWARD: Forward ``join_asof`` from ``aligned_caldt`` within each ``kypermno``.
        EXACT_ON_ALIGNED_DATE: Exact equality join on ``(kypermno, aligned_caldt)``.
    """

    ASOF_FORWARD = "ASOF_FORWARD"
    EXACT_ON_ALIGNED_DATE = "EXACT_ON_ALIGNED_DATE"


@dataclass(frozen=True)
class SecCcmJoinSpecV1:
    """Versioned join specification persisted with SEC-CCM pre-merge artifacts.

    Attributes:
        version: Join-spec schema version tag.
        alignment_policy: Legacy Phase B alignment policy selector.
        timezone: Time zone used for acceptance-datetime interpretation.
        daily_join_enabled: Whether Phase B daily join is enabled.
        daily_join_source: Source family for daily rows.
        daily_permno_col: Input column name for PERMNO/KYPERMNO in daily input.
        daily_date_col: Input column name for calendar date in daily input.
        daily_feature_columns: Daily feature columns projected into Phase B join output.
        required_daily_non_null_features: Features that must be non-null for a row to be usable.
    """

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
        """Serialize this join specification to a plain dictionary.

        Returns:
            dict: Dataclass fields as JSON-serializable key/value pairs.
        """
        return asdict(self)

    def write_json(self, out_path: Path) -> Path:
        """Write this join specification to disk as formatted JSON.

        Args:
            out_path: Output JSON path.

        Returns:
            Path: The resolved output path that was written.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out_path


@dataclass(frozen=True)
class SecCcmJoinSpecV2:
    """Versioned join specification with explicit Phase B alignment/join modes.

    Attributes:
        version: Join-spec schema version tag.
        phase_b_alignment_mode: Date-alignment strategy used before daily joining.
        phase_b_daily_join_mode: Daily join strategy used after alignment.
        timezone: Time zone used for acceptance-datetime interpretation.
        daily_join_enabled: Whether Phase B daily join is enabled.
        daily_join_source: Source family for daily rows.
        daily_permno_col: Input column name for PERMNO/KYPERMNO in daily input.
        daily_date_col: Input column name for calendar date in daily input.
        daily_feature_columns: Daily feature columns projected into Phase B join output.
        required_daily_non_null_features: Features that must be non-null for a row to be usable.
        daily_join_max_forward_lag_days: Maximum allowed forward lag days for
            ``ASOF_FORWARD`` daily joins. ``None`` disables lag gating.
    """

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
        """Serialize this join specification to a plain dictionary.

        Enum members are emitted as their string ``value`` fields.

        Returns:
            dict: Dataclass fields as JSON-serializable key/value pairs.
        """
        payload = asdict(self)
        payload["phase_b_alignment_mode"] = self.phase_b_alignment_mode.value
        payload["phase_b_daily_join_mode"] = self.phase_b_daily_join_mode.value
        return payload

    def write_json(self, out_path: Path) -> Path:
        """Write this join specification to disk as formatted JSON.

        Args:
            out_path: Output JSON path.

        Returns:
            Path: The output path that was written.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out_path


SecCcmJoinSpec = SecCcmJoinSpecV1 | SecCcmJoinSpecV2


def normalize_sec_ccm_join_spec(join_spec: SecCcmJoinSpec) -> SecCcmJoinSpecV2:
    """Normalize a V1 or V2 join specification to canonical V2.

    WHY (TODO by Erik): Persisting one canonical spec shape avoids downstream
    branching when reports and artifact manifests are compared across runs.

    Args:
        join_spec: Input join specification.

    Returns:
        SecCcmJoinSpecV2: Normalized V2 join specification.

    Raises:
        TypeError: If ``join_spec`` is neither ``SecCcmJoinSpecV1`` nor ``SecCcmJoinSpecV2``.
        ValueError: If V1 policy ``FIRST_CLOSE_AFTER_ACCEPTANCE`` is requested.
    """
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
    """Build a canonical V2 join-spec preset.

    Args:
        preset_name: Preset identifier.
        base: Optional seed spec whose non-policy fields are preserved.

    Returns:
        SecCcmJoinSpecV2: Preset join specification.

    Raises:
        ValueError: If ``preset_name`` is unknown.
    """
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
