from __future__ import annotations

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
