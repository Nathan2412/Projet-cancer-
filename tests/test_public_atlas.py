import gzip
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import import_genie_atlas
import import_public_atlas
from allele_atlas import AlleleAtlas
from validate_public_atlas import evaluate_transfer, feature_rows


def test_api_pagination_and_cohort_filters(monkeypatch):
    monkeypatch.setattr(import_public_atlas, "PAGE_SIZE", 2)
    samples = ["s1", "s2", "s3"]
    labels = [
        {"sampleId": sample, "patientId": subject, "clinicalAttributeId": "CANCER_TYPE", "value": cancer}
        for sample, subject, cancer in (("s1", "p1", "Melanoma"), ("s2", "p1", "Melanoma"), ("s3", "p2", "Breast Cancer"))
    ]
    mutations = [
        {"sampleId": sample, "patientId": subject, "entrezGeneId": 673, "proteinChange": protein, "mutationStatus": status}
        for sample, subject, protein, status in (
            ("s1", "p1", "V600E", "Somatic"), ("s2", "p1", "V600E", "NA"), ("s3", "p2", "V600K", "Germline"),
        )
    ]

    def api(path, *, params=None, body=None):
        if path.endswith("/sample-lists"):
            return [{"sampleListId": "study_seq", "category": "all_cases_with_mutation_data"}]
        if path.endswith("/molecular-profiles"):
            return [{"molecularProfileId": "study_mut", "molecularAlterationType": "MUTATION_EXTENDED"}]
        if path.endswith("/sample-ids"):
            return samples
        offset = params["pageSize"] * params["pageNumber"]
        rows = labels if path.endswith("/clinical-data") else [row for row in mutations if row["sampleId"] in body["sampleIds"]]
        return rows[offset:offset + params["pageSize"]]

    monkeypatch.setattr(import_public_atlas, "_api", api)
    with ThreadPoolExecutor(max_workers=1) as pool:
        atlas, report = import_public_atlas._study_data("study", {"sequencedSampleCount": 3}, pool, None)
    assert report["patients"] == 2
    assert report["labelled_samples"] == 3
    assert report["call_statuses"] == {"explicit_somatic": 1, "unspecified": 1, "excluded_other_status": 1}
    assert atlas.payload(source="cohort")["alleles"] == {"BRAF V600E": {"subjects": 1, "cancers": {"Melanoma": 1}}}
    with ThreadPoolExecutor(max_workers=1) as pool:
        complement, filtered = import_public_atlas._study_data(
            "study", {"sequencedSampleCount": 3}, pool, None, exclude_subjects={"p1"},
        )
    assert set(complement.groups["Breast Cancer"]) == {"p2"}
    assert filtered["excluded_previously_seen_samples"] == 2


def test_genie_join_gzip_and_deduplication(tmp_path):
    clinical = tmp_path / "clinical.tsv"
    clinical.write_text(
        "# cBioPortal metadata\nPATIENT_ID\tSAMPLE_ID\tCANCER_TYPE\n"
        "P1\tS1\tMelanoma\nP1\tS2\tMelanoma\nP2\tS3\tBreast Cancer\n", encoding="utf-8",
    )
    mutations = tmp_path / "mutations.tsv.gz"
    with gzip.open(mutations, "wt", encoding="utf-8") as handle:
        handle.write(
            "Tumor_Sample_Barcode\tHugo_Symbol\tHGVSp_Short\tMutation_Status\n"
            "S1\tBRAF\tp.V600E\tSomatic\nS2\tBRAF\tp.V600E\tSomatic\nS3\tTP53\tp.R175H\tGermline\n",
        )
    output = tmp_path / "atlas.json"
    import_genie_atlas.build_genie_atlas(clinical, mutations, "20.0-public", output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["subjects"] == 2
    assert payload["sequenced_samples"] == 3
    assert payload["alleles"] == {"BRAF V600E": {"subjects": 1, "cancers": {"Melanoma": 1}}}
    assert '"P1"' not in output.read_text(encoding="utf-8")
    with gzip.open(mutations, "at", encoding="utf-8") as handle:
        handle.write("S4\tBRAF\tp.V600E\tSomatic\n")
    with pytest.raises(ValueError, match="même version"):
        import_genie_atlas.build_genie_atlas(clinical, mutations, "20.0-public", output=output)


def test_external_validation_uses_distinct_sources_and_excludes_conflicts():
    train = AlleleAtlas()
    test = AlleleAtlas()
    for atlas, prefix in ((train, "train"), (test, "test")):
        for index in range(8):
            subject = f"{prefix}-{index}"
            label = "Melanoma" if index % 2 else "Pancreatic Cancer"
            atlas.add_patient(label, subject)
            atlas.add_variant(*(("BRAF", "V600E") if index % 2 else ("KRAS", "G12D")), label, subject)
    train.add_patient("Melanoma", "conflict")
    train.add_patient("Pancreatic Cancer", "conflict")
    _, labels, conflicts = feature_rows(train)
    assert len(labels) == 8 and conflicts == 1
    result = evaluate_transfer(train, test)
    assert result["training_patients"] == result["test_patients"] == 8
    assert result["models"]["Régression logistique"]["f1_macro"] == 1
    assert result["models"]["Classe majoritaire"]["top3_accuracy"] is None
