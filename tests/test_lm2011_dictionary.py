from __future__ import annotations

import json
from pathlib import Path

from thesis_pkg.core.sec.lm2011_dictionary import (
    EXTENDED_DICTIONARY_FAMILY_NAME,
    GENERATED_DICTIONARY_FAMILIES_DIRNAME,
    HARVARD_NEGATIVE_WORD_LIST_FILE,
    REPLICATION_DICTIONARY_FAMILY_NAME,
    load_lm2011_dictionary_inputs,
    materialize_lm2011_dictionary_families,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_dictionary_source_dir(tmp_path: Path) -> Path:
    additional_data_dir = tmp_path / "LM2011_additional_data"
    _write_text(additional_data_dir / HARVARD_NEGATIVE_WORD_LIST_FILE, "HARVARD_ONE\nHARVARD_TWO\n")
    _write_text(
        additional_data_dir / "LM2011_MasterDictionary.txt",
        "\n".join(
            [
                "Word,Negative,Positive,Uncertainty,Litigious,Modal,Source",
                "LOSS,2009,0,0,0,1,12of12inf",
                "GAIN,0,2009,0,0,0,10K_2008",
                "RISK,0,0,2009,0,3,10K_2009",
                "LAWSUIT,0,0,0,2009,0,12of12inf",
                "STALE_NEG,2011,0,0,0,0,10K_2010",
            ]
        ),
    )
    _write_text(
        additional_data_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv",
        "\n".join(
            [
                "Word,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining,Complexity,Source",
                "LOSS,2009,0,0,0,2009,0,0,0,12of12inf",
                "GAIN,0,2012,0,0,0,0,0,0,10K_2008",
                "RISK,0,0,2011,0,0,2011,0,0,10K_2009",
                "LAWSUIT,0,0,0,2014,0,0,2014,0,10K_2009",
                "BOUND,0,0,0,0,0,0,2018,0,10K_2010",
                "DENSE,0,0,0,0,0,0,0,2024,10K_2010",
                "REMOVED_NEG,-2020,0,0,0,0,0,0,0,10K_2010",
                "ZEROES,0,0,0,0,0,0,0,0,10K_2010",
            ]
        ),
    )
    return additional_data_dir


def test_materialize_dictionary_families_exports_loader_compatible_replication_bundle(tmp_path: Path) -> None:
    additional_data_dir = _build_dictionary_source_dir(tmp_path)

    generated = materialize_lm2011_dictionary_families(additional_data_dir)

    assert generated.root_dir == additional_data_dir / GENERATED_DICTIONARY_FAMILIES_DIRNAME
    assert generated.replication.family_name == REPLICATION_DICTIONARY_FAMILY_NAME
    assert generated.extended.family_name == EXTENDED_DICTIONARY_FAMILY_NAME
    assert generated.replication.directory.exists()
    assert generated.extended.directory.exists()
    assert generated.replication.manifest_path.exists()
    assert generated.extended.manifest_path.exists()

    for filename in (
        "Fin-Neg.txt",
        "Fin-Pos.txt",
        "Fin-Unc.txt",
        "Fin-Lit.txt",
        "MW-Strong.txt",
        "MW-Weak.txt",
        HARVARD_NEGATIVE_WORD_LIST_FILE,
        "LM2011_MasterDictionary.txt",
    ):
        assert (generated.replication.directory / filename).exists()

    dictionary_inputs = load_lm2011_dictionary_inputs(generated.replication.directory)
    assert dictionary_inputs.dictionary_lists["negative"] == ("loss",)
    assert dictionary_inputs.dictionary_lists["positive"] == ("gain",)
    assert dictionary_inputs.dictionary_lists["uncertainty"] == ("risk",)
    assert dictionary_inputs.dictionary_lists["litigious"] == ("lawsuit",)
    assert dictionary_inputs.dictionary_lists["modal_strong"] == ("loss",)
    assert dictionary_inputs.dictionary_lists["modal_weak"] == ("risk",)
    assert dictionary_inputs.harvard_negative_word_list == ("harvard_one", "harvard_two")
    assert dictionary_inputs.master_dictionary_words == ("LOSS", "GAIN", "RISK", "LAWSUIT")

    replication_manifest = json.loads(generated.replication.manifest_path.read_text(encoding="utf-8"))
    assert replication_manifest["dictionary_lists"]["negative"]["word_count"] == 1
    assert replication_manifest["dictionary_lists"]["positive"]["word_count"] == 1
    assert replication_manifest["dictionary_lists"]["uncertainty"]["word_count"] == 1
    assert replication_manifest["dictionary_lists"]["litigious"]["word_count"] == 1
    assert replication_manifest["dictionary_lists"]["modal_strong"]["word_count"] == 1
    assert replication_manifest["dictionary_lists"]["modal_weak"]["word_count"] == 1
    assert replication_manifest["dictionary_lists"]["harvard_negative"]["word_count"] == 2
    assert replication_manifest["master_dictionary"]["word_count"] == 4


def test_materialize_dictionary_families_exports_active_extended_bundle_and_excludes_removed_entries(
    tmp_path: Path,
) -> None:
    additional_data_dir = _build_dictionary_source_dir(tmp_path)

    generated = materialize_lm2011_dictionary_families(additional_data_dir)

    for filename in (
        "Fin-Neg.txt",
        "Fin-Pos.txt",
        "Fin-Unc.txt",
        "Fin-Lit.txt",
        "MW-Strong.txt",
        "MW-Weak.txt",
        "Constraining.txt",
        "Complexity.txt",
        HARVARD_NEGATIVE_WORD_LIST_FILE,
        "Loughran-McDonald_MasterDictionary_1993-2024.csv",
    ):
        assert (generated.extended.directory / filename).exists()

    dictionary_inputs = load_lm2011_dictionary_inputs(generated.extended.directory)
    assert dictionary_inputs.dictionary_lists["negative"] == ("loss",)
    assert dictionary_inputs.dictionary_lists["positive"] == ("gain",)
    assert dictionary_inputs.dictionary_lists["uncertainty"] == ("risk",)
    assert dictionary_inputs.dictionary_lists["litigious"] == ("lawsuit",)
    assert dictionary_inputs.dictionary_lists["modal_strong"] == ("loss",)
    assert dictionary_inputs.dictionary_lists["modal_weak"] == ("risk",)
    assert "removed_neg" not in dictionary_inputs.dictionary_lists["negative"]
    assert "zeroes" not in dictionary_inputs.dictionary_lists["negative"]
    assert dictionary_inputs.master_dictionary_words == (
        "LOSS",
        "GAIN",
        "RISK",
        "LAWSUIT",
        "BOUND",
        "DENSE",
        "REMOVED_NEG",
        "ZEROES",
    )

    extended_manifest = json.loads(generated.extended.manifest_path.read_text(encoding="utf-8"))
    assert extended_manifest["dictionary_lists"]["negative"]["word_count"] == 1
    assert extended_manifest["dictionary_lists"]["positive"]["word_count"] == 1
    assert extended_manifest["dictionary_lists"]["uncertainty"]["word_count"] == 1
    assert extended_manifest["dictionary_lists"]["litigious"]["word_count"] == 1
    assert extended_manifest["dictionary_lists"]["modal_strong"]["word_count"] == 1
    assert extended_manifest["dictionary_lists"]["modal_weak"]["word_count"] == 1
    assert extended_manifest["dictionary_lists"]["constraining"]["word_count"] == 2
    assert extended_manifest["dictionary_lists"]["complexity"]["word_count"] == 1
    assert extended_manifest["master_dictionary"]["word_count"] == 8
