"""Tests unitaires pour annotator.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from annotator import (
    compute_pathogenicity_score,
    classify_variant,
    annotate_gene_mutations,
    annotate_with_known_db,
    _matches_hotspot,
    _infer_significance,
)


# ── compute_pathogenicity_score ───────────────────────────────────────────────

def test_pathogenicity_high_impact():
    mut = {"impact": "HIGH", "frequency": 0.5, "known": False}
    score = compute_pathogenicity_score(mut)
    assert score >= 0.4


def test_pathogenicity_known_mutation():
    mut = {"impact": "MODERATE", "frequency": 0.1, "known": True}
    score = compute_pathogenicity_score(mut)
    assert score >= 0.5


def test_pathogenicity_capped_at_1():
    mut = {
        "impact": "HIGH", "frequency": 0.9, "known": True,
        "associated_cancers": ["A", "B", "C"],
        "type": "DEL", "length": 4,  # frameshift (4 % 3 != 0)
    }
    score = compute_pathogenicity_score(mut)
    assert score == 1.0


def test_pathogenicity_benign():
    mut = {"impact": "LOW", "frequency": 0.01, "known": False}
    score = compute_pathogenicity_score(mut)
    assert score < 0.3


# ── classify_variant ──────────────────────────────────────────────────────────

def test_classify_pathogenic():
    mut = {"impact": "HIGH", "frequency": 0.9, "known": True,
           "associated_cancers": ["Sein", "Ovaire", "Pancreas"]}
    cls, score = classify_variant(mut)
    assert cls == "Pathogenic"
    assert score >= 0.8


def test_classify_vus():
    mut = {"impact": "MODERATE", "frequency": 0.05, "known": False, "associated_cancers": []}
    cls, score = classify_variant(mut)
    assert cls in ("VUS", "Likely_pathogenic", "Likely_benign")


def test_classify_benign():
    mut = {"impact": "MODIFIER", "frequency": 0.0, "known": False, "associated_cancers": []}
    cls, score = classify_variant(mut)
    assert cls in ("Benign", "Likely_benign")


# ── _matches_hotspot ──────────────────────────────────────────────────────────

def test_matches_hotspot_exact_protein_change():
    mut = {"protein_change": "R175H"}
    hotspot = {"change": "R175H", "cancers": ["Sein"], "frequency": 0.05}
    assert _matches_hotspot(mut, hotspot) is True


def test_matches_hotspot_case_insensitive():
    mut = {"protein_change": "r175h"}
    hotspot = {"change": "R175H", "cancers": ["Sein"], "frequency": 0.05}
    assert _matches_hotspot(mut, hotspot) is True


def test_matches_hotspot_no_match():
    mut = {"protein_change": "V600E"}
    hotspot = {"change": "R175H", "cancers": ["Sein"], "frequency": 0.05}
    assert _matches_hotspot(mut, hotspot) is False


def test_matches_hotspot_no_protein_change():
    mut = {"position": 175}
    hotspot = {"change": "R175H", "cancers": ["Sein"], "frequency": 0.05}
    assert _matches_hotspot(mut, hotspot) is False


def test_matches_hotspot_via_hotspot_change():
    mut = {"hotspot_change": "V600E"}
    hotspot = {"change": "V600E", "cancers": ["Melanome"], "frequency": 0.50}
    assert _matches_hotspot(mut, hotspot) is True


# ── annotate_with_known_db ────────────────────────────────────────────────────

def _make_known_db():
    return {
        "TP53": {
            "hotspots": [
                {"change": "R175H", "cancers": ["Sein", "Poumon"], "frequency": 0.05},
                {"change": "R248W", "cancers": ["Colon"], "frequency": 0.03},
            ]
        }
    }


def test_annotate_known_mutation():
    known_db = _make_known_db()
    mutations = [{"protein_change": "R175H", "impact": "HIGH", "frequency": 0.4}]
    annotated = annotate_with_known_db(mutations, known_db, "TP53")
    assert len(annotated) == 1
    a = annotated[0]
    assert a["known"] is True
    assert a["is_hotspot"] is True
    assert a["clinical_significance"] == "Pathogenic"
    assert "Sein" in a["associated_cancers"]


def test_annotate_unknown_mutation():
    known_db = _make_known_db()
    mutations = [{"protein_change": "G245S", "impact": "HIGH", "frequency": 0.1}]
    annotated = annotate_with_known_db(mutations, known_db, "TP53")
    a = annotated[0]
    assert a["known"] is False
    assert a["is_hotspot"] is False
    assert a["clinical_significance"] in ("Likely_pathogenic", "VUS", "Likely_benign", "Benign")


def test_annotate_gene_not_in_db():
    mutations = [{"protein_change": "X100Y", "impact": "LOW", "frequency": 0.1}]
    annotated = annotate_with_known_db(mutations, {}, "UNKNOWNGENE")
    assert len(annotated) == 1
    assert annotated[0]["known"] is False


def test_annotate_empty_mutations():
    known_db = _make_known_db()
    annotated = annotate_with_known_db([], known_db, "TP53")
    assert annotated == []


# ── annotate_gene_mutations (integration) ────────────────────────────────────

def test_annotate_gene_mutations_adds_pathogenicity():
    known_db = _make_known_db()
    mutations = [{"protein_change": "R175H", "impact": "HIGH", "frequency": 0.4, "gene": "TP53"}]
    result = annotate_gene_mutations(mutations, known_db, "TP53")
    assert "pathogenicity_score" in result[0]
    assert "acmg_classification" in result[0]
    assert result[0]["pathogenicity_score"] > 0


def test_annotate_gene_mutations_adds_gene_info():
    known_db = _make_known_db()
    mutations = [{"protein_change": "R175H", "impact": "HIGH", "frequency": 0.4, "gene": "TP53"}]
    result = annotate_gene_mutations(mutations, known_db, "TP53")
    assert "gene_description" in result[0]
    assert "chromosome" in result[0]


# ── _infer_significance ───────────────────────────────────────────────────────

def test_infer_high_impact():
    assert _infer_significance({"impact": "HIGH"}) == "Likely_pathogenic"


def test_infer_moderate_high_freq():
    assert _infer_significance({"impact": "MODERATE", "frequency": 0.4}) == "Likely_pathogenic"


def test_infer_moderate_low_freq():
    assert _infer_significance({"impact": "MODERATE", "frequency": 0.1}) == "VUS"


def test_infer_low_impact():
    assert _infer_significance({"impact": "LOW"}) == "Likely_benign"


def test_infer_modifier():
    assert _infer_significance({"impact": "MODIFIER"}) == "VUS"
