import json

import pytest

import allele_atlas
import import_msk_atlas


def test_aggregate_deduplicates_subject_and_preserves_multitumor_rows(tmp_path):
    atlas = allele_atlas.AlleleAtlas()
    for cancer, subject in [
        ("Mélanome", "one"), ("Mélanome", "two"), ("Thyroïde", "one"), ("Mélanome", "one"),
    ]:
        atlas.add_patient(cancer, subject)
        atlas.add_variant("braf", "p.Val600Glu", cancer, subject)
        atlas.add_variant("BRAF", "v600e", cancer, subject)
    path = atlas.export(tmp_path / "index.json", source="Test")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["subjects"] == 2
    assert data["alleles"]["BRAF V600E"] == {"subjects": 2, "cancers": {"Mélanome": 2, "Thyroïde": 1}}
    assert data["cancers"] == {"Mélanome": 2, "Thyroïde": 1}
    assert '"one"' not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(("query", "expected"), [
    ("braf p.Val600Glu", ["BRAF V600E"]),
    ("V600E", ["BRAF V600E"]),
    ("BRAF V600D", []),
    ("BRAF V600", ["BRAF V600E", "BRAF V600K"]),
    ("BRAF", ["BRAF V600E", "BRAF V600K"]),
    ("BRAF p.Val600Met", []),
])
def test_search_distinguishes_variants(query, expected):
    data = {"alleles": {"BRAF V600E": {}, "BRAF V600K": {}, "KRAS G12D": {}}}
    assert allele_atlas.matching_names(data, query) == expected


def test_msk_joins_by_sample_and_keeps_only_somatic_calls(monkeypatch, tmp_path):
    target = tmp_path / "msk.json"
    monkeypatch.setitem(import_msk_atlas.ATLAS_PATHS, "MSK-IMPACT 50K", target)
    ids = ["A1", "A2", "B1", "Unlabelled"]
    labels = [("A1", "SubA", "Melanoma"), ("A2", "SubA", "Thyroid Cancer"), ("B1", "SubB", "Melanoma")]

    def fake_api(path, *, params=None, body=None):
        if "sample-lists" in path:
            return ids
        if "clinical-data" in path:
            return [
                {"clinicalAttributeId": "CANCER_TYPE", "sampleId": sample, "patientId": subject, "value": cancer}
                for sample, subject, cancer in labels
            ]
        if "molecular-profiles" in path:
            assert body["sampleIds"] == ["A1", "A2", "B1"]
            return [
                {"sampleId": sample, "entrezGeneId": 673, "proteinChange": "V600E", "mutationStatus": status}
                for sample, status in [("A1", "SOMATIC"), ("A2", "SOMATIC"), ("B1", "GERMLINE")]
            ]
        return {"name": "MSK test", "sequencedSampleCount": len(ids)}

    monkeypatch.setattr(import_msk_atlas, "_api", fake_api)
    import_msk_atlas.build_msk_atlas()
    atlas = json.loads(target.read_text(encoding="utf-8"))
    assert atlas["subjects"] == 2
    assert atlas["missing_labels"] == 1
    assert atlas["alleles"]["BRAF V600E"] == {"subjects": 1, "cancers": {"Melanoma": 1, "Thyroid Cancer": 1}}


def test_msk_extension_updates_added_genes_without_recounting_cohort():
    base = allele_atlas.AlleleAtlas()
    extension = allele_atlas.AlleleAtlas()
    for atlas in (base, extension):
        atlas.add_patient("Melanoma", "patient")
    base.add_variant("BRAF", "V600E", "Melanoma", "patient")
    base.add_variant("NRAS", "Q61K", "Melanoma", "patient")
    extension.add_variant("NRAS", "Q61R", "Melanoma", "patient")
    metadata = {"source": "MSK", "source_url": allele_atlas.MSK_STUDY_URL, "sequenced_samples": 1}

    merged = import_msk_atlas.merge_panel(base.payload(**metadata), extension.payload(**metadata))

    assert merged["subjects"] == 1
    assert set(merged["alleles"]) == {"BRAF V600E", "NRAS Q61R"}
    with pytest.raises(ValueError, match="ne correspondent pas"):
        import_msk_atlas.merge_panel(base.payload(**metadata), dict(merged, subjects=2))
