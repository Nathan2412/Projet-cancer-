"""Tests de non-regression pour les regles cliniques et la CLI."""

import sys

from clinical_rules import normalize_cancer_label, sex_cancer_status
from download_real_data import _stable_gene_seed, build_dataset_manifest
from loader import validate_metadata
from main import configure_logging, parse_args
from ml_predictor import apply_sex_constraints, _prediction_call


def test_normalize_cancer_label():
    assert normalize_cancer_label("breast cancer") == "Sein"
    assert normalize_cancer_label("thyroid carcinoma") == "Thyroide"


def test_sex_cancer_status_allows_rare_male_breast_case():
    assert sex_cancer_status("M", "Sein") == "rare"


def test_sex_cancer_status_excludes_female_prostate_case():
    assert sex_cancer_status("F", "Prostate") == "excluded"


def test_validate_metadata_warns_but_keeps_rare_case():
    patients = [{
        "patient_id": "PAT_0001",
        "metadata": {"sex": "M", "cancer_type": "Sein", "age": 52},
    }]
    report = validate_metadata(patients, verbose=False)
    assert report["n_excluded"] == 0
    assert any("Cas rare mais possible" in msg for msg in report["warnings"])


def test_apply_sex_constraints_excludes_impossible_and_penalizes_rare():
    probs = {"Sein": 0.5, "Prostate": 0.3, "Colon": 0.2}
    constrained = apply_sex_constraints(probs, "M")
    assert constrained["Prostate"] > 0
    assert constrained["Sein"] < 0.5

    probs_f = {"Prostate": 0.5, "Colon": 0.5}
    constrained_f = apply_sex_constraints(probs_f, "F")
    assert constrained_f["Prostate"] == 0.0
    assert constrained_f["Colon"] == 1.0


def test_parse_args_supports_real_data_alias(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--real-data", "--quiet"])
    args = parse_args()
    assert args.real_data is True
    assert args.mode == "real"
    assert args.quiet is True


def test_configure_logging_runs_without_error():
    configure_logging()


def test_stable_gene_seed_is_deterministic():
    assert _stable_gene_seed("TP53") == _stable_gene_seed("TP53")
    assert _stable_gene_seed("TP53") != _stable_gene_seed("BRCA1")


def test_build_dataset_manifest_aggregates_studies():
    cohort = [
        {"original_study": "study_a", "cancer_type": "Sein", "severity": "high"},
        {"original_study": "study_a", "cancer_type": "Sein", "severity": "high"},
        {"original_study": "study_b", "cancer_type": "Colon", "severity": "low"},
    ]
    manifest = build_dataset_manifest(cohort)
    assert manifest["total_patients"] == 3
    assert manifest["studies"]["study_a"]["n_patients"] == 2
    assert manifest["studies"]["study_b"]["cancer_types"]["Colon"] == 1


def test_prediction_call_marks_low_confidence_as_uncertain():
    label, is_uncertain = _prediction_call("Colon", 0.2)
    assert label == "Prediction incertaine"
    assert is_uncertain is True


def test_prediction_call_keeps_high_confidence_label():
    label, is_uncertain = _prediction_call("Colon", 0.8)
    assert label == "Colon"
    assert is_uncertain is False
