from collections import Counter

import retrain_multicohort as retrain
from allele_atlas import AlleleAtlas


def _atlas(*entries):
    atlas = AlleleAtlas()
    for subject, label, gene, variant in entries:
        atlas.add_patient(label, subject)
        atlas.add_variant(gene, variant, label, subject)
    return atlas


def test_profiles_keep_variants_in_their_tumor_and_reject_conflicts():
    atlas = _atlas(
        ("one", "Breast Cancer", "PIK3CA", "H1047R"),
        ("one", "Breast Carcinoma", "TP53", "R175H"),
        ("two", "Melanoma", "BRAF", "V600E"),
        ("two", "Unknown Primary", "TP53", "R175H"),
    )
    profiles, counts = retrain.patient_profiles(atlas)
    assert profiles == {"one": ("Cancer du sein", {"PIK3CA H1047R", "TP53 R175H"})}
    assert counts == {"available": 2, "eligible": 1, "conflicts": 1, "unmapped": 0}


def test_patient_deduplication_is_limited_to_comparable_identifiers():
    records, origins, conflicts = {}, {}, set()
    breast = _atlas(("one", "Breast Cancer", "PIK3CA", "H1047R"))
    extra = _atlas(("one", "Breast Carcinoma", "TP53", "R175H"))
    discordant = _atlas(("one", "Melanoma", "BRAF", "V600E"))
    for source, name in ((breast, "first"), (extra, "second")):
        retrain.record_source(records, origins, conflicts, source, "msk", name)
    assert records[("msk", "one")] == ("Cancer du sein", {"PIK3CA H1047R", "TP53 R175H"})
    assert Counter(origins.values()) == {"first": 1}
    retrain.record_source(records, origins, conflicts, discordant, "msk", "discordant")
    retrain.record_source(records, origins, conflicts, breast, "msk", "repeat")
    retrain.record_source(records, origins, conflicts, breast, "gdc", "independent-ID-system")
    assert set(records) == {("gdc", "one")}
    assert conflicts == {("msk", "one")}


def test_same_test_set_and_training_only_vocabulary(monkeypatch):
    monkeypatch.setattr(retrain, "CLASSES", ["Sein", "Thyroïde"])
    training = [("Sein", {"TP53 R175H"}), ("Thyroïde", {"BRAF V600E"})] * 4
    testing = [
        ("Sein", {"TP53 R175H", "TEST_ONLY"}),
        ("Thyroïde", {"BRAF V600E", "TEST_ONLY"}),
    ]
    result = retrain.fit_and_score({
        "Classe majoritaire": training, "Régression de référence": training,
        "Régression élargie": training * 2,
    }, testing)
    assert result["Régression élargie"]["features"] == 2
    assert result["Régression élargie"]["training_patients"] == 16
    assert result["Régression élargie"]["f1_macro"] == 1
    assert all(sum(row["support"] for row in model["per_class"]) == 2 for model in result.values())
