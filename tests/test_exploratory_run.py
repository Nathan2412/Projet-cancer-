import json

import pytest

import run_results
from src.ml import exploration


def _sample(index, cancer, subject):
    patient_id = f"PAT_{index:04}"
    return {
        "patient_id": patient_id,
        "metadata": {"cancer_type": cancer, "original_patient_id": subject},
        "gene_analyses": {},
        "total_mutations_detected": index % 3,
    }


def test_grouped_validation_uses_only_training_subjects(monkeypatch):
    results = [
        _sample(index, "Colon" if index < 6 else "Poumon", f"subject-{index // 2}")
        for index in range(12)
    ]
    matrix = {row["patient_id"]: {"TP53": row["total_mutations_detected"]} for row in results}
    train_sizes = []

    def signatures(training_rows):
        train_sizes.append(len(training_rows))
        subjects = {row["metadata"]["original_patient_id"] for row in training_rows}
        assert len(training_rows) == 2 * len(subjects)
        return {}

    monkeypatch.setattr(exploration, "build_cancer_allele_signatures", signatures)
    out = exploration.evaluate_exploratory(results, ["TP53"], matrix)
    assert len(train_sizes) == 3
    assert out["patients"] == 12
    assert len(out["models"]) == 3
    for model in out["models"].values():
        assert sum(map(sum, model["confusion_matrix"])) == 12
        assert model["top3_accuracy"] is None


def test_limited_cohort_has_no_model_for_single_class():
    rows = [_sample(index, "Colon", f"subject-{index}") for index in range(4)]
    out = exploration.evaluate_exploratory(rows, [], {})
    assert out["models"] == {}
    assert "unavailable" in out


def test_snapshot_is_aggregate_and_failed_write_keeps_previous(tmp_path, monkeypatch):
    target = tmp_path / "reports" / "summary.json"
    monkeypatch.setattr(run_results, "SUMMARY_PATH", target)
    row = _sample(1, "Colon", "PRIVATE_SOURCE_ID")
    matrix = {row["patient_id"]: {"TP53": 1}}
    timings = {"preparation_seconds": 1, "elapsed_seconds": 2, "is_limited": True}
    run_results.publish_run_summary([row], ["TP53"], matrix, {}, **timings)
    initial = target.read_text(encoding="utf-8")
    assert "PRIVATE_SOURCE_ID" not in initial
    assert row["patient_id"] not in initial
    assert json.loads(initial)["genes"] == [{"gene": "TP53", "patients": 1, "variants": 1}]
    with pytest.raises(ValueError):
        run_results.publish_run_summary([row], ["TP53"], matrix, {"bad": float("nan")}, **timings)
    assert target.read_text(encoding="utf-8") == initial
    assert list(target.parent.iterdir()) == [target]
