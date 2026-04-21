from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


REPO_LOCAL_OPERATIVE_INPUT = "repo-local operative replication input"
UPDATED_MASTER_DICTIONARY_INPUT = (
    "updated master-dictionary input available in the local data directory; "
    "not provenance-verified as the original LM2011 paper-distributed source"
)
UPDATED_NON_PAPER_ERA_RESOURCE = "updated non-paper-era master-dictionary resource"

LM2011_OPERATIVE_WORD_LIST_FILES: dict[str, str] = {
    "negative": "Fin-Neg.txt",
    "positive": "Fin-Pos.txt",
    "uncertainty": "Fin-Unc.txt",
    "litigious": "Fin-Lit.txt",
    "modal_strong": "MW-Strong.txt",
    "modal_weak": "MW-Weak.txt",
}
HARVARD_NEGATIVE_WORD_LIST_FILE = "Harvard_IV_NEG_Inf.txt"
MASTER_DICTIONARY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("LM2011_MasterDictionary.txt", UPDATED_MASTER_DICTIONARY_INPUT),
    ("Loughran-McDonald_MasterDictionary_1993-2024.csv", UPDATED_NON_PAPER_ERA_RESOURCE),
)
GENERATED_DICTIONARY_FAMILIES_DIRNAME = "generated_dictionary_families"
REPLICATION_DICTIONARY_FAMILY_NAME = "replication"
EXTENDED_DICTIONARY_FAMILY_NAME = "extended"
PAPER_ERA_SENTIMENT_YEAR = "2009"
PAPER_ERA_MASTER_SOURCES: tuple[str, ...] = ("12of12inf", "10K_2008", "10K_2009")


@dataclass(frozen=True)
class _GeneratedListSelectionRule:
    category: str
    source_column: str
    output_filename: str
    selected_value: str | None = None
    required: bool = True


REPLICATION_LIST_SELECTION_RULES: tuple[_GeneratedListSelectionRule, ...] = (
    _GeneratedListSelectionRule("negative", "Negative", "Fin-Neg.txt", selected_value=PAPER_ERA_SENTIMENT_YEAR),
    _GeneratedListSelectionRule("positive", "Positive", "Fin-Pos.txt", selected_value=PAPER_ERA_SENTIMENT_YEAR),
    _GeneratedListSelectionRule("uncertainty", "Uncertainty", "Fin-Unc.txt", selected_value=PAPER_ERA_SENTIMENT_YEAR),
    _GeneratedListSelectionRule("litigious", "Litigious", "Fin-Lit.txt", selected_value=PAPER_ERA_SENTIMENT_YEAR),
    _GeneratedListSelectionRule("modal_strong", "Modal", "MW-Strong.txt", selected_value="1"),
    _GeneratedListSelectionRule("modal_weak", "Modal", "MW-Weak.txt", selected_value="3"),
)
EXTENDED_LIST_SELECTION_RULES: tuple[_GeneratedListSelectionRule, ...] = (
    _GeneratedListSelectionRule("negative", "Negative", "Fin-Neg.txt"),
    _GeneratedListSelectionRule("positive", "Positive", "Fin-Pos.txt"),
    _GeneratedListSelectionRule("uncertainty", "Uncertainty", "Fin-Unc.txt"),
    _GeneratedListSelectionRule("litigious", "Litigious", "Fin-Lit.txt"),
    _GeneratedListSelectionRule("modal_strong", "Strong_Modal", "MW-Strong.txt"),
    _GeneratedListSelectionRule("modal_weak", "Weak_Modal", "MW-Weak.txt"),
    _GeneratedListSelectionRule("constraining", "Constraining", "Constraining.txt", required=False),
    _GeneratedListSelectionRule("complexity", "Complexity", "Complexity.txt", required=False),
)


@dataclass(frozen=True)
class Lm2011GeneratedDictionaryFamily:
    family_name: str
    directory: Path
    manifest_path: Path
    master_dictionary_path: Path
    master_dictionary_word_count: int
    dictionary_list_paths: dict[str, Path]
    dictionary_list_counts: dict[str, int]
    source_resources: dict[str, str]
    selection_rules: dict[str, Any]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "directory": str(self.directory.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "source_resources": dict(self.source_resources),
            "selection_rules": dict(self.selection_rules),
            "dictionary_lists": {
                category: {
                    "path": str(path.resolve()),
                    "word_count": self.dictionary_list_counts[category],
                }
                for category, path in self.dictionary_list_paths.items()
            },
            "master_dictionary": {
                "path": str(self.master_dictionary_path.resolve()),
                "word_count": self.master_dictionary_word_count,
            },
        }


@dataclass(frozen=True)
class Lm2011GeneratedDictionaryFamilies:
    root_dir: Path
    replication: Lm2011GeneratedDictionaryFamily
    extended: Lm2011GeneratedDictionaryFamily

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir.resolve()),
            "families": {
                self.replication.family_name: self.replication.to_manifest_dict(),
                self.extended.family_name: self.extended.to_manifest_dict(),
            },
        }


@dataclass(frozen=True)
class Lm2011LexiconResource:
    name: str
    path: Path
    role: str
    provenance_status: str
    word_count: int
    file_size_bytes: int

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path.resolve()),
            "role": self.role,
            "provenance_status": self.provenance_status,
            "word_count": self.word_count,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass(frozen=True)
class Lm2011DictionaryInputs:
    dictionary_lists: dict[str, tuple[str, ...]]
    harvard_negative_word_list: tuple[str, ...]
    master_dictionary_words: tuple[str, ...]
    resources: tuple[Lm2011LexiconResource, ...]
    master_dictionary_path: Path
    master_dictionary_provenance_status: str

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "resource_scope": "repo-local operative LM2011-style lexicon inputs",
            "historical_provenance_warning": (
                "These paths are the repo-local operative inputs for this run. "
                "The selected master dictionary is not asserted to be the original "
                "LM2011 paper-distributed dictionary unless separately verified."
            ),
            "master_dictionary_path": str(self.master_dictionary_path.resolve()),
            "master_dictionary_provenance_status": self.master_dictionary_provenance_status,
            "resources": [resource.to_manifest_dict() for resource in self.resources],
        }


def _require_existing_nonempty_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def _normalize_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _require_csv_columns(
    *,
    fieldnames: list[str] | None,
    required_columns: tuple[str, ...],
    label: str,
) -> list[str]:
    available = list(fieldnames or ())
    missing = [name for name in required_columns if name not in available]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    return available


def _read_csv_rows(
    path: Path,
    *,
    label: str,
    required_columns: tuple[str, ...],
) -> tuple[list[str], list[dict[str, str]]]:
    resolved_path = _require_existing_nonempty_path(path, label=label)
    with resolved_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = _require_csv_columns(
            fieldnames=reader.fieldnames,
            required_columns=required_columns,
            label=label,
        )
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{label} contained no data rows: {resolved_path}")
    return fieldnames, rows


def _write_word_list(path: Path, words: list[str]) -> None:
    path.write_text("\n".join(words) + "\n", encoding="utf-8")


def _copy_word_list(source_path: Path, destination_path: Path) -> int:
    source_words = load_lm2011_word_list(source_path)
    shutil.copy2(source_path, destination_path)
    return len(source_words)


def _write_csv_rows(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_equal_match_words(
    rows: list[dict[str, str]],
    *,
    column: str,
    selected_value: str,
) -> list[str]:
    selected_words = [
        _normalize_cell(row.get("Word"))
        for row in rows
        if _normalize_cell(row.get(column)) == selected_value and _normalize_cell(row.get("Word"))
    ]
    return _unique_preserve_order(selected_words)


def _is_active_extended_membership(value: str | None) -> bool:
    normalized = _normalize_cell(value)
    if not normalized or normalized == "0":
        return False
    return not normalized.startswith("-")


def _select_active_words(
    rows: list[dict[str, str]],
    *,
    column: str,
) -> list[str]:
    selected_words = [
        _normalize_cell(row.get("Word"))
        for row in rows
        if _normalize_cell(row.get("Word")) and _is_active_extended_membership(row.get(column))
    ]
    return _unique_preserve_order(selected_words)


def _reset_generated_dictionary_family_root(root_dir: Path) -> None:
    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)


def _write_family_manifest(
    family_dir: Path,
    *,
    payload: dict[str, Any],
) -> Path:
    manifest_path = family_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _materialize_replication_dictionary_family(
    base_dir: Path,
    family_dir: Path,
) -> Lm2011GeneratedDictionaryFamily:
    source_master_dictionary_path = base_dir / "LM2011_MasterDictionary.txt"
    fieldnames, rows = _read_csv_rows(
        source_master_dictionary_path,
        label="LM2011 master dictionary file",
        required_columns=("Word", "Negative", "Positive", "Uncertainty", "Litigious", "Modal", "Source"),
    )
    family_dir.mkdir(parents=True, exist_ok=True)

    dictionary_list_paths: dict[str, Path] = {}
    dictionary_list_counts: dict[str, int] = {}
    for rule in REPLICATION_LIST_SELECTION_RULES:
        words = _select_equal_match_words(
            rows,
            column=rule.source_column,
            selected_value=rule.selected_value or "",
        )
        output_path = family_dir / rule.output_filename
        _write_word_list(output_path, words)
        dictionary_list_paths[rule.category] = output_path
        dictionary_list_counts[rule.category] = len(words)

    harvard_source_path = _require_existing_nonempty_path(
        base_dir / HARVARD_NEGATIVE_WORD_LIST_FILE,
        label="Harvard negative word list",
    )
    harvard_output_path = family_dir / HARVARD_NEGATIVE_WORD_LIST_FILE
    dictionary_list_paths["harvard_negative"] = harvard_output_path
    dictionary_list_counts["harvard_negative"] = _copy_word_list(harvard_source_path, harvard_output_path)

    filtered_rows = [
        row for row in rows if _normalize_cell(row.get("Source")) in PAPER_ERA_MASTER_SOURCES
    ]
    master_dictionary_output_path = family_dir / "LM2011_MasterDictionary.txt"
    _write_csv_rows(master_dictionary_output_path, fieldnames=fieldnames, rows=filtered_rows)

    selection_rules = {
        "dictionary_lists": {
            rule.category: {
                "source_column": rule.source_column,
                "selected_value": rule.selected_value,
            }
            for rule in REPLICATION_LIST_SELECTION_RULES
        },
        "recognized_word_master_dictionary": {
            "source_column": "Source",
            "allowed_source_values": list(PAPER_ERA_MASTER_SOURCES),
        },
        "copied_resources": {
            "harvard_negative": HARVARD_NEGATIVE_WORD_LIST_FILE,
        },
    }
    manifest_path = _write_family_manifest(
        family_dir,
        payload={
            "family_name": REPLICATION_DICTIONARY_FAMILY_NAME,
            "source_resources": {
                "master_dictionary": str(source_master_dictionary_path.resolve()),
                "harvard_negative": str(harvard_source_path.resolve()),
            },
            "selection_rules": selection_rules,
            "dictionary_lists": {
                category: {
                    "path": str(path.resolve()),
                    "word_count": dictionary_list_counts[category],
                }
                for category, path in dictionary_list_paths.items()
            },
            "master_dictionary": {
                "path": str(master_dictionary_output_path.resolve()),
                "word_count": len(filtered_rows),
            },
        },
    )
    return Lm2011GeneratedDictionaryFamily(
        family_name=REPLICATION_DICTIONARY_FAMILY_NAME,
        directory=family_dir,
        manifest_path=manifest_path,
        master_dictionary_path=master_dictionary_output_path,
        master_dictionary_word_count=len(filtered_rows),
        dictionary_list_paths=dictionary_list_paths,
        dictionary_list_counts=dictionary_list_counts,
        source_resources={
            "master_dictionary": str(source_master_dictionary_path.resolve()),
            "harvard_negative": str(harvard_source_path.resolve()),
        },
        selection_rules=selection_rules,
    )


def _materialize_extended_dictionary_family(
    base_dir: Path,
    family_dir: Path,
) -> Lm2011GeneratedDictionaryFamily:
    source_master_dictionary_path = base_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
    fieldnames, rows = _read_csv_rows(
        source_master_dictionary_path,
        label="extended LM master dictionary file",
        required_columns=("Word", "Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal", "Weak_Modal"),
    )
    family_dir.mkdir(parents=True, exist_ok=True)

    dictionary_list_paths: dict[str, Path] = {}
    dictionary_list_counts: dict[str, int] = {}
    available_columns = set(fieldnames)
    for rule in EXTENDED_LIST_SELECTION_RULES:
        if rule.source_column not in available_columns:
            if rule.required:
                raise ValueError(
                    f"Extended LM master dictionary file is missing required column {rule.source_column!r}: "
                    f"{source_master_dictionary_path}"
                )
            continue
        words = _select_active_words(rows, column=rule.source_column)
        if not words and not rule.required:
            continue
        output_path = family_dir / rule.output_filename
        _write_word_list(output_path, words)
        dictionary_list_paths[rule.category] = output_path
        dictionary_list_counts[rule.category] = len(words)

    harvard_source_path = _require_existing_nonempty_path(
        base_dir / HARVARD_NEGATIVE_WORD_LIST_FILE,
        label="Harvard negative word list",
    )
    harvard_output_path = family_dir / HARVARD_NEGATIVE_WORD_LIST_FILE
    dictionary_list_paths["harvard_negative"] = harvard_output_path
    dictionary_list_counts["harvard_negative"] = _copy_word_list(harvard_source_path, harvard_output_path)

    master_dictionary_output_path = family_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
    shutil.copy2(source_master_dictionary_path, master_dictionary_output_path)

    selection_rules = {
        "dictionary_lists": {
            rule.category: {
                "source_column": rule.source_column,
                "selected_membership_rule": "positive_year_flag_only",
            }
            for rule in EXTENDED_LIST_SELECTION_RULES
            if rule.required or rule.source_column in available_columns
        },
        "recognized_word_master_dictionary": {
            "selection_rule": "full_current_csv_word_universe",
        },
        "copied_resources": {
            "harvard_negative": HARVARD_NEGATIVE_WORD_LIST_FILE,
        },
    }
    manifest_path = _write_family_manifest(
        family_dir,
        payload={
            "family_name": EXTENDED_DICTIONARY_FAMILY_NAME,
            "source_resources": {
                "master_dictionary": str(source_master_dictionary_path.resolve()),
                "harvard_negative": str(harvard_source_path.resolve()),
            },
            "selection_rules": selection_rules,
            "dictionary_lists": {
                category: {
                    "path": str(path.resolve()),
                    "word_count": dictionary_list_counts[category],
                }
                for category, path in dictionary_list_paths.items()
            },
            "master_dictionary": {
                "path": str(master_dictionary_output_path.resolve()),
                "word_count": len(rows),
            },
        },
    )
    return Lm2011GeneratedDictionaryFamily(
        family_name=EXTENDED_DICTIONARY_FAMILY_NAME,
        directory=family_dir,
        manifest_path=manifest_path,
        master_dictionary_path=master_dictionary_output_path,
        master_dictionary_word_count=len(rows),
        dictionary_list_paths=dictionary_list_paths,
        dictionary_list_counts=dictionary_list_counts,
        source_resources={
            "master_dictionary": str(source_master_dictionary_path.resolve()),
            "harvard_negative": str(harvard_source_path.resolve()),
        },
        selection_rules=selection_rules,
    )


def materialize_lm2011_dictionary_families(additional_data_dir: Path | str) -> Lm2011GeneratedDictionaryFamilies:
    base_dir = Path(additional_data_dir)
    generated_root = base_dir / GENERATED_DICTIONARY_FAMILIES_DIRNAME
    _reset_generated_dictionary_family_root(generated_root)
    replication_family = _materialize_replication_dictionary_family(
        base_dir,
        generated_root / REPLICATION_DICTIONARY_FAMILY_NAME,
    )
    extended_family = _materialize_extended_dictionary_family(
        base_dir,
        generated_root / EXTENDED_DICTIONARY_FAMILY_NAME,
    )
    return Lm2011GeneratedDictionaryFamilies(
        root_dir=generated_root,
        replication=replication_family,
        extended=extended_family,
    )


def load_lm2011_word_list(path: Path | str) -> tuple[str, ...]:
    resolved_path = _require_existing_nonempty_path(Path(path), label="LM2011 word-list file")
    words: list[str] = []
    with resolved_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            token = line.strip()
            if token:
                words.append(token.casefold())
    if not words:
        raise ValueError(f"LM2011 word-list file contained no tokens: {resolved_path}")
    return tuple(words)


def resolve_lm2011_master_dictionary_path(additional_data_dir: Path | str) -> tuple[Path, str]:
    base_dir = Path(additional_data_dir)
    for filename, provenance_status in MASTER_DICTIONARY_CANDIDATES:
        candidate = base_dir / filename
        if candidate.exists():
            return _require_existing_nonempty_path(candidate, label="LM master dictionary file"), provenance_status
    expected = [filename for filename, _ in MASTER_DICTIONARY_CANDIDATES]
    raise FileNotFoundError(f"No LM master dictionary file found in {base_dir}; expected one of {expected}")


def load_lm2011_master_dictionary_words(additional_data_dir: Path | str) -> tuple[tuple[str, ...], Path, str]:
    dictionary_path, provenance_status = resolve_lm2011_master_dictionary_path(additional_data_dir)
    words_df = pl.read_csv(
        dictionary_path,
        schema_overrides={"Word": pl.Utf8},
        infer_schema_length=0,
    )
    if "Word" not in words_df.columns:
        raise ValueError(f"LM master dictionary file must contain a 'Word' column: {dictionary_path}")
    words = tuple(
        word.strip()
        for word in words_df.get_column("Word").drop_nulls().to_list()
        if isinstance(word, str) and word.strip()
    )
    if not words:
        raise ValueError(f"LM master dictionary file contained no usable Word values: {dictionary_path}")
    return words, dictionary_path, provenance_status


def load_lm2011_dictionary_inputs(additional_data_dir: Path | str) -> Lm2011DictionaryInputs:
    base_dir = Path(additional_data_dir)
    dictionary_lists: dict[str, tuple[str, ...]] = {}
    resources: list[Lm2011LexiconResource] = []

    for category, filename in LM2011_OPERATIVE_WORD_LIST_FILES.items():
        path = _require_existing_nonempty_path(base_dir / filename, label=f"LM2011 {category} word list")
        words = load_lm2011_word_list(path)
        dictionary_lists[category] = words
        resources.append(
            Lm2011LexiconResource(
                name=filename,
                path=path,
                role=f"dictionary_list:{category}",
                provenance_status=REPO_LOCAL_OPERATIVE_INPUT,
                word_count=len(words),
                file_size_bytes=path.stat().st_size,
            )
        )

    harvard_path = _require_existing_nonempty_path(
        base_dir / HARVARD_NEGATIVE_WORD_LIST_FILE,
        label="Harvard negative word list",
    )
    harvard_negative_word_list = load_lm2011_word_list(harvard_path)
    resources.append(
        Lm2011LexiconResource(
            name=HARVARD_NEGATIVE_WORD_LIST_FILE,
            path=harvard_path,
            role="dictionary_list:harvard_negative",
            provenance_status=REPO_LOCAL_OPERATIVE_INPUT,
            word_count=len(harvard_negative_word_list),
            file_size_bytes=harvard_path.stat().st_size,
        )
    )

    master_dictionary_words, master_dictionary_path, master_dictionary_provenance_status = (
        load_lm2011_master_dictionary_words(base_dir)
    )
    resources.append(
        Lm2011LexiconResource(
            name=master_dictionary_path.name,
            path=master_dictionary_path,
            role="recognized_word_master_dictionary",
            provenance_status=master_dictionary_provenance_status,
            word_count=len(master_dictionary_words),
            file_size_bytes=master_dictionary_path.stat().st_size,
        )
    )

    return Lm2011DictionaryInputs(
        dictionary_lists=dictionary_lists,
        harvard_negative_word_list=harvard_negative_word_list,
        master_dictionary_words=master_dictionary_words,
        resources=tuple(resources),
        master_dictionary_path=master_dictionary_path,
        master_dictionary_provenance_status=master_dictionary_provenance_status,
    )
