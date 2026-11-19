"""Import cBioPortal par cohorte, sans conserver d’identifiants dans les résultats."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError

from allele_atlas import AlleleAtlas
from cohort_catalog import COHORTS, FIXED_LABELS, OPTIONAL_COHORTS, PORTAL, TCGA, cohort_path
from download_real_data import ENTREZ_TO_GENE
from import_msk_atlas import EXTRA_GENES, PAGE_SIZE, _api

PANEL = ENTREZ_TO_GENE | EXTRA_GENES
BATCH_SIZE = 400
UNKNOWN = {"", "na", "n/a", "not reported", "unknown", "[not available]", "not available"}
SOMATIC = {"SOMATIC", "LIKELY SOMATIC"}
UNSPECIFIED = {"", ".", "NA", "N/A", "UNKNOWN", "NOT REPORTED"}


def _study_catalog():
    rows = _api("studies", params={"projection": "DETAILED", "pageSize": 9999})
    return {row["studyId"]: row for row in rows}


def _choose(items, id_field, preferred):
    exact = next((item[id_field] for item in items if item[id_field] == preferred), None)
    if exact:
        return exact
    if len(items) == 1:
        return items[0][id_field]
    raise ValueError(f"Sélection ambiguë ou absente : {preferred}.")


def _paged(path, params, *, body=None):
    page = 0
    first_key = None
    while True:
        rows = _api(path, params={**params, "pageSize": PAGE_SIZE, "pageNumber": page}, body=body)
        if not isinstance(rows, list):
            raise ValueError(f"Réponse non tabulaire : {path}.")
        if rows:
            current_key = rows[0]
            if current_key == first_key:
                raise ValueError(f"Page répétée : {path}.")
            first_key = current_key
        yield from rows
        if len(rows) < PAGE_SIZE:
            return
        page += 1


def _label_map(study_id, sampled, fixed_label):
    if fixed_label:
        rows = _paged(f"studies/{study_id}/samples", {"projection": "SUMMARY"})
    else:
        rows = _paged(f"studies/{study_id}/clinical-data", {"clinicalDataType": "SAMPLE", "attributeId": "CANCER_TYPE"})
    labels = {}
    for row in rows:
        if not fixed_label and row["clinicalAttributeId"] != "CANCER_TYPE":
            raise ValueError("Filtre clinique CANCER_TYPE non appliqué.")
        sample = row["sampleId"]
        if sample not in sampled:
            continue
        subject = str(row.get("patientId") or "").strip()
        label = str(fixed_label or row.get("value") or "").strip()
        if not subject or label.casefold() in UNKNOWN:
            continue
        if sample in labels and labels[sample] != (subject, label):
            raise ValueError("Étiquette clinique contradictoire pour un échantillon.")
        labels[sample] = (subject, label)
    return labels


def _fetch_mutations(profile, sample_ids):
    path = f"molecular-profiles/{profile}/mutations/fetch"
    body = {"sampleIds": sample_ids, "entrezGeneIds": list(PANEL)}
    return list(_paged(path, {"projection": "SUMMARY"}, body=body))


def _study_data(study_id, study, pool, fixed_label, *, require_explicit_somatic=False, exclude_subjects=None):
    lists = _api(f"studies/{study_id}/sample-lists")
    profiles = _api(f"studies/{study_id}/molecular-profiles")
    sample_list = _choose(
        [item for item in lists if item.get("category") == "all_cases_with_mutation_data"
         or item["sampleListId"] == f"{study_id}_sequenced"],
        "sampleListId", f"{study_id}_sequenced",
    )
    profile = _choose(
        [item for item in profiles if item.get("molecularAlterationType") == "MUTATION_EXTENDED"],
        "molecularProfileId", f"{study_id}_mutations",
    )
    sample_ids = _api(f"sample-lists/{sample_list}/sample-ids")
    sampled = set(sample_ids)
    if not sampled or len(sampled) != len(sample_ids):
        raise ValueError("Liste d’échantillons vide ou avec des doublons.")
    reported = study.get("sequencedSampleCount")
    labels = _label_map(study_id, sampled, fixed_label)
    excluded_samples = 0
    if exclude_subjects is not None:
        before = len(labels)
        labels = {sample: pair for sample, pair in labels.items() if pair[0] not in exclude_subjects}
        excluded_samples = before - len(labels)
    if not labels:
        raise ValueError("Aucun échantillon séquencé avec un type tumoral exploitable.")
    ordered = sorted(labels)
    batches = [ordered[start:start + BATCH_SIZE] for start in range(0, len(ordered), BATCH_SIZE)]
    statuses = Counter()
    atlas = AlleleAtlas()
    for subject, cancer in labels.values():
        atlas.add_patient(cancer, subject)
    for batch, mutations in zip(batches, pool.map(lambda batch: _fetch_mutations(profile, batch), batches), strict=True):
        requested = set(batch)
        for row in mutations:
            sample = row["sampleId"]
            if sample not in requested:
                raise ValueError("Mutation hors du lot demandé.")
            subject, cancer = labels[sample]
            if row.get("patientId") and row["patientId"] != subject:
                raise ValueError("Les données cliniques et mutationnelles ne concordent pas.")
            status = str(row.get("mutationStatus") or "").strip().upper()
            if status not in SOMATIC and (require_explicit_somatic or status not in UNSPECIFIED):
                statuses["excluded_other_status"] += 1
                continue
            statuses["explicit_somatic" if status in SOMATIC else "unspecified"] += 1
            atlas.add_variant(PANEL.get(row["entrezGeneId"]), row.get("proteinChange"), cancer, subject)
    return atlas, {
        "study_id": study_id,
        "sequenced_samples": len(sampled),
        "reported_sequenced_samples": reported,
        "labelled_samples": len(labels),
        "excluded_previously_seen_samples": excluded_samples,
        "patients": len(set().union(*atlas.groups.values())),
        "variants": len(atlas.carriers),
        "call_statuses": dict(statuses),
    }


def _merge_atlas(target, source):
    for cancer, subjects in source.groups.items():
        target.groups[cancer].update(subjects)
    for allele, carriers in source.carriers.items():
        for cancer, subjects in carriers.items():
            target.carriers[allele][cancer].update(subjects)


def build_public_atlas(key, studies, *, jobs=3, path=None):
    label, study_id = COHORTS[key]
    ids = [study_id] if study_id else sorted(name for name in studies if name.endswith("_tcga_pan_can_atlas_2018"))
    if not ids:
        raise ValueError("Aucune étude TCGA PanCancer Atlas.")
    target = AlleleAtlas()
    metadata = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for identifier in ids:
            cohort, study = _study_data(identifier, studies[identifier], pool, FIXED_LABELS.get(key))
            _merge_atlas(target, cohort)
            metadata.append(study)
            print(f"{identifier}: {study['patients']:,} patients · {study['variants']:,} variants", flush=True)
    if not target.carriers:
        raise ValueError("Aucun variant protéique dans le panel.")
    samples = sum(row["sequenced_samples"] for row in metadata)
    return target.export(
        path or cohort_path(key), source=label, source_url=f"{PORTAL}{study_id}" if study_id else TCGA,
        sequenced_samples=samples, missing_labels=samples - sum(row["labelled_samples"] for row in metadata),
        provenance=metadata,
    )


def main():
    parser = argparse.ArgumentParser(description="Construire les catalogues tumoraux publics (cBioPortal).")
    parser.add_argument("sources", nargs="+", metavar="SOURCE", help=f"all ou {', '.join(COHORTS)}")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--directory", type=Path, help="Dossier de sortie optionnel pour une reconstruction isolée.")
    args = parser.parse_args()
    keys = [key for key in COHORTS if key not in OPTIONAL_COHORTS] if args.sources == ["all"] else list(dict.fromkeys(args.sources))
    if unknown := set(keys) - COHORTS.keys():
        parser.error(f"Sources inconnues : {', '.join(sorted(unknown))}.")
    studies = _study_catalog()
    failures = []
    for key in keys:
        path = args.directory / cohort_path(key).name if args.directory else None
        try:
            print(build_public_atlas(key, studies, jobs=args.jobs, path=path), flush=True)
        except (URLError, OSError, ValueError, KeyError) as exc:
            failures.append(key)
            print(f"ÉCHEC {key} : {exc}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
