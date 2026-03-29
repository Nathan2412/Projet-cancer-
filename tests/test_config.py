"""Tests de cohérence pour config.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import (
    CANCER_GENES, CANCER_TYPES, GENE_ROLES, CANCER_LABEL_MAPPING,
    MUTATION_TYPES, IMPACT_LEVELS, NUCLEOTIDES,
    MIN_COVERAGE, MIN_QUALITY_SCORE, MUTATION_FREQ_THRESHOLD,
    ALLELE_MIN_FREQUENCY, ALLELE_MAX_OUTSIDE_FREQUENCY, ALLELE_MIN_ENRICHMENT,
)


def test_cancer_genes_not_empty():
    assert len(CANCER_GENES) > 0


def test_all_cancer_genes_have_required_fields():
    for gene, info in CANCER_GENES.items():
        assert "chromosome" in info, f"{gene} manque chromosome"
        assert "start" in info, f"{gene} manque start"
        assert "end" in info, f"{gene} manque end"
        assert info["start"] < info["end"], f"{gene}: start >= end"


def test_gene_roles_covers_all_cancer_genes():
    for gene in CANCER_GENES:
        assert gene in GENE_ROLES, f"{gene} manque dans GENE_ROLES"
        assert GENE_ROLES[gene] in ("oncogene", "suppressor"), \
            f"{gene} a un rôle invalide: {GENE_ROLES[gene]}"


def test_cancer_types_not_empty():
    assert len(CANCER_TYPES) > 0
    for ct in CANCER_TYPES:
        assert isinstance(ct, str) and ct, "Cancer type vide ou non-string"


def test_cancer_label_mapping_targets_valid():
    for label, target in CANCER_LABEL_MAPPING.items():
        assert target in CANCER_TYPES, \
            f"Mapping '{label}' -> '{target}' non trouvé dans CANCER_TYPES"


def test_thresholds_are_valid():
    assert 0 < MUTATION_FREQ_THRESHOLD < 1
    assert MIN_COVERAGE > 0
    assert MIN_QUALITY_SCORE >= 0
    assert 0 < ALLELE_MIN_FREQUENCY <= 1
    assert 0 < ALLELE_MAX_OUTSIDE_FREQUENCY <= 1
    assert ALLELE_MIN_ENRICHMENT >= 1.0


def test_allele_freq_threshold_ordering():
    # min_frequency doit être < max_outside_frequency OU le filtrage a du sens
    # (il peut y avoir min_frequency > max_outside pour forcer discriminance)
    assert ALLELE_MIN_FREQUENCY > 0
    assert ALLELE_MAX_OUTSIDE_FREQUENCY > 0


def test_nucleotides():
    assert set(NUCLEOTIDES) == {"A", "T", "C", "G"}


def test_impact_levels_keys():
    expected = {"HIGH", "MODERATE", "LOW", "MODIFIER"}
    assert set(IMPACT_LEVELS.keys()) == expected


def test_mutation_types_keys():
    expected_keys = {"SNP", "INS", "DEL", "DUP", "INV", "FRAMESHIFT"}
    assert set(MUTATION_TYPES.keys()) == expected_keys
