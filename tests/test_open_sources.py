import gzip
import io
import json

import import_gdc_atlas
import import_icgc_atlas


def test_gdc_filters_to_open_cases_and_canonical_mutations(monkeypatch, tmp_path):
    project = "CCDI-MCI"

    def case(subject, site, diagnosis):
        return {"case_id": subject, "project": {"project_id": project}, "primary_site": site, "disease_type": diagnosis}

    first = case("patient-1", "Brain", "Gliomas")
    second = case("patient-2", "Brain", "Gliomas")
    unknown = case("patient-3", "Not Reported", "Unknown")
    duplicate = case("patient-4", "Skin", "Melanoma")

    def occurrence(study_case, change, canonical):
        return {"case": study_case, "ssm": {"consequence": [
            {"transcript": {"gene": {"symbol": "BRAF"}, "is_canonical": canonical, "aa_change": change}},
        ]}}

    def rows(endpoint, filters, _fields, _sort):
        if endpoint == "files":
            assert import_gdc_atlas._term("access", "open") in filters
            return [
                {"access": "open", "cases": [first, second, unknown]},
                {"access": "open", "cases": [first]},
            ]
        return [
            occurrence(first, "V600E", True), occurrence(first, "V600E", True),
            occurrence(second, "V600K", False), occurrence(duplicate, "V600E", True),
        ]

    monkeypatch.setattr(import_gdc_atlas, "_rows", rows)
    target = tmp_path / "atlas.json"
    import_gdc_atlas.build_gdc_atlas(project, release="Release 46", path=target)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["subjects"] == 2
    assert data["missing_labels"] == 1
    assert data["alleles"] == {"BRAF V600E": {"subjects": 1, "cancers": {"Brain · Gliomas": 1}}}
    assert "patient-1" not in target.read_text(encoding="utf-8")


def test_icgc_canonical_substitutions_from_public_fragments():
    handle = io.BytesIO(gzip.compress(
        b"DO123\tBLCA-CN\tmissense_variant\tV600E\tENSG1.1\tENST1.2\textra\n"
        b"DO123\tBLCA-CN\tmissense_variant\tV600E\tENSG1.1\tENST1.2\textra\n"
        b"DO123\tBLCA-CN\tmissense_variant\tV567E\tENSG1\tENST2\textra\n"
        b"DO123\tBLCA-CN\tsynonymous_variant\tV600V\tENSG1\tENST1\textra\n"
    ))
    result = import_icgc_atlas.parse_fragment(handle, tuple(range(6)), "BLCA-CN", "DO123", {"ENSG1": ("BRAF", "ENST1")})
    assert result == ("DO123", True, {("BRAF", "V600E")})


def test_gdc_braf_hotspot_uses_genomic_identity_not_just_protein_label():
    row = {"ssm": {"genomic_dna_change": "chr7:g.140753336A>T", "consequence": [
        {"transcript": {"gene": {"symbol": "BRAF"}, "is_canonical": True, "aa_change": "V640E"}},
    ]}}
    assert list(import_gdc_atlas.protein_variants(row)) == [("BRAF", "V600E")]
    del row["ssm"]["genomic_dna_change"]
    assert list(import_gdc_atlas.protein_variants(row)) == [("BRAF", "V640E")]


def test_icgc_pagination_keeps_only_open_mutation_objects(monkeypatch):
    responses = iter([
        b"<R><Contents><Key>release_28/data/BLCA-CN/DO123/ssm_open/part-00000.gz</Key></Contents>"
        b"<Contents><Key>release_28/data/BLCA-CN/DO123/donor/part-00000.gz</Key></Contents>"
        b"<IsTruncated>true</IsTruncated><NextContinuationToken>next+1</NextContinuationToken></R>",
        b"<R><Contents><Key>release_28/data/BLCA-CN/DO456/ssm_open/part-00003.gz</Key></Contents>"
        b"<IsTruncated>false</IsTruncated></R>",
    ])
    urls = []

    def read(url, parser):
        urls.append(url)
        return parser(io.BytesIO(next(responses)))

    monkeypatch.setattr(import_icgc_atlas, "_read_object", read)
    assert [donor for _, donor in import_icgc_atlas.list_mutation_objects("BLCA-CN")] == ["DO123", "DO456"]
    assert "continuation-token=next%2B1" in urls[1]
