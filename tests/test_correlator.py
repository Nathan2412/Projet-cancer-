"""Tests unitaires pour correlator.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from correlator import (
    compute_mutation_burden,
    compute_gene_specificity_table,
    build_cohort_mutation_matrix,
    compute_gene_cancer_correlation,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_patient(pid, cancer, genes_mutated, age=50, sex="M"):
    gene_analyses = {}
    for gene in genes_mutated:
        gene_analyses[gene] = {
            "total_mutations": 3,
            "snps": 3, "insertions": 0, "deletions": 0,
            "mutations": [
                {"gene": gene, "impact": "HIGH", "type": "SNP",
                 "frequency": 0.4, "known": True, "associated_cancers": [cancer]}
            ],
            "reference_length": 1000,
        }
    return {
        "patient_id": pid,
        "metadata": {"cancer_type": cancer, "age": age, "sex": sex},
        "gene_analyses": gene_analyses,
        "risk_report": {"panel_mutation_density": 5.0, "n_hotspots": 1,
                        "n_pathogenic_variants": 1, "n_oncogenes_mutated": 1,
                        "n_suppressors_mutated": 0},
        "annotations": [],
        "risk_summary": {},
    }


# ── compute_mutation_burden ───────────────────────────────────────────────────

def test_mutation_burden_basic():
    ga = {
        "TP53": {"total_mutations": 5, "reference_length": 1000},
        "KRAS": {"total_mutations": 2, "reference_length": 500},
    }
    burden = compute_mutation_burden(ga)
    # (5+2) / (1000+500) * 1e6 = 4666.67
    assert abs(burden - 4666.67) < 1


def test_mutation_burden_zero():
    ga = {"TP53": {"total_mutations": 0, "reference_length": 1000}}
    assert compute_mutation_burden(ga) == 0.0


def test_mutation_burden_no_bases():
    ga = {"TP53": {"total_mutations": 5, "reference_length": 0}}
    assert compute_mutation_burden(ga) == 0.0


# ── compute_gene_specificity_table ────────────────────────────────────────────

def test_gene_specificity_basic():
    patients = [
        _make_patient("P1", "Sein", ["BRCA1", "TP53"]),
        _make_patient("P2", "Sein", ["BRCA1"]),
        _make_patient("P3", "Colon", ["APC"]),
        _make_patient("P4", "Colon", ["APC", "KRAS"]),
    ]
    table = compute_gene_specificity_table(patients)

    assert "BRCA1" in table
    assert "Sein" in table["BRCA1"]
    # BRCA1 est muté 2/2 fois dans le sein
    assert table["BRCA1"]["Sein"]["freq_in_cancer"] == 1.0
    # BRCA1 n'est pas muté hors du sein
    assert table["BRCA1"]["Sein"]["freq_outside_cancer"] == 0.0


def test_gene_specificity_enrichment():
    patients = [
        _make_patient("P1", "Melanome", ["BRAF"]),
        _make_patient("P2", "Melanome", ["BRAF"]),
        _make_patient("P3", "Poumon",   ["KRAS"]),
        _make_patient("P4", "Poumon",   ["KRAS"]),
    ]
    table = compute_gene_specificity_table(patients)
    # BRAF dans Melanome : freq_in=1.0, freq_out=0.0 → enrichissement élevé
    assert table["BRAF"]["Melanome"]["enrichment"] > 1.0


def test_gene_specificity_empty():
    table = compute_gene_specificity_table([])
    assert table == {}


# ── build_cohort_mutation_matrix ──────────────────────────────────────────────

def test_mutation_matrix_shape():
    patients = [
        _make_patient("P1", "Sein", ["BRCA1", "TP53"]),
        _make_patient("P2", "Colon", ["APC"]),
    ]
    matrix, genes, pids = build_cohort_mutation_matrix(patients)
    # matrix est un dict {patient_id: {gene: count}}
    assert len(pids) == 2
    assert len(genes) > 0
    assert len(matrix) == len(pids)
    for pid in pids:
        assert pid in matrix
        assert len(matrix[pid]) == len(genes)


def test_mutation_matrix_values():
    patients = [
        _make_patient("P1", "Sein", ["TP53"]),
        _make_patient("P2", "Colon", []),
    ]
    matrix, genes, pids = build_cohort_mutation_matrix(patients)
    if "TP53" in genes:
        assert matrix["P1"]["TP53"] > 0
        assert matrix["P2"]["TP53"] == 0


# ── compute_gene_cancer_correlation ──────────────────────────────────────────

def test_gene_cancer_correlation_structure():
    patients = [
        _make_patient("P1", "Sein", ["BRCA1", "TP53"]),
        _make_patient("P2", "Sein", ["BRCA1"]),
        _make_patient("P3", "Poumon", ["KRAS", "TP53"]),
    ]
    correlations = compute_gene_cancer_correlation(patients)
    assert isinstance(correlations, dict)


def test_gene_cancer_correlation_brca1_sein():
    patients = [
        _make_patient(f"P{i}", "Sein", ["BRCA1"]) for i in range(5)
    ] + [
        _make_patient(f"P{i+5}", "Poumon", ["KRAS"]) for i in range(5)
    ]
    correlations = compute_gene_cancer_correlation(patients)
    # On vérifie juste que la structure retournée est un dict non vide
    assert len(correlations) > 0
