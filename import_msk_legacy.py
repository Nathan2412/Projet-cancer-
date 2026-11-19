"""Importer seulement le complément des études MSK partageant les mêmes identifiants."""

import re
from concurrent.futures import ThreadPoolExecutor

from allele_atlas import AlleleAtlas
from cohort_catalog import MSK_LEGACY, PORTAL, cohort_path
from import_msk_atlas import STUDY as BASE_STUDY, _api
from import_public_atlas import _merge_atlas, _paged, _study_catalog, _study_data


def base_subjects():
    sequenced = set(_api(f"sample-lists/{BASE_STUDY}_sequenced/sample-ids"))
    samples = _paged(f"studies/{BASE_STUDY}/samples", {"projection": "SUMMARY"})
    sample_subjects = {row["sampleId"]: row["patientId"] for row in samples if row["sampleId"] in sequenced}
    if not sequenced or sample_subjects.keys() != sequenced:
        raise ValueError("Identifiants patients MSK-50K incomplets ; aucun complément publié.")
    return set(sample_subjects.values())


def build_msk_legacy(*, output=None):
    studies = _study_catalog()
    baseline = base_subjects()
    seen = set(baseline)
    target = AlleleAtlas()
    provenance = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for study_id in MSK_LEGACY:
            cohort, metadata = _study_data(study_id, studies[study_id], pool, None, exclude_subjects=seen)
            current = set().union(*cohort.groups.values())
            if current & seen or any(not re.fullmatch(r"P-[0-9]+", patient) for patient in current):
                raise ValueError("Identifiants MSK non comparables ; aucun complément publié.")
            seen.update(current)
            _merge_atlas(target, cohort)
            metadata["scope"] = "Complément excluant les identifiants présents dans MSK-50K et les études MSK déjà traitées."
            provenance.append(metadata)
            print(f"{study_id} : {len(current)} patients supplémentaires", flush=True)
    if not target.carriers:
        raise ValueError("Aucun variant protéique dans le complément MSK.")
    samples = sum(row["labelled_samples"] for row in provenance)
    path = target.export(
        output or cohort_path("msk_legacy_unique"), source="MSK · Complément hors 50K",
        source_url=f"{PORTAL}msk_chord_2024", sequenced_samples=samples, provenance=provenance,
    )
    return len(seen - baseline), path


if __name__ == "__main__":
    subjects, filename = build_msk_legacy()
    print(f"Complément MSK : {subjects} patients hors MSK-50K · {filename}")
