"""Indexer localement deux fichiers obtenus par l’accès officiel AACR Project GENIE."""

import argparse
import csv
import gzip
import re
from collections import Counter
from pathlib import Path

from allele_atlas import AlleleAtlas
from cohort_catalog import GENIE, cohort_path
from import_public_atlas import PANEL, SOMATIC, UNKNOWN, UNSPECIFIED


def _table(path, required):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        if missing := required - set(reader.fieldnames or []):
            raise ValueError(f"Champs manquants dans {path.name} : {', '.join(sorted(missing))}")
        yield from reader


def build_genie_atlas(clinical_file, mutation_file, release, *, output=None):
    if not re.fullmatch(r"[0-9]+\.[0-9]+-public", release):
        raise ValueError("Version du registre attendue, par exemple 20.0-public.")
    panel = set(PANEL.values())
    labels = {}
    sample_ids = set()
    atlas = AlleleAtlas()
    for row in _table(clinical_file, {"PATIENT_ID", "SAMPLE_ID", "CANCER_TYPE"}):
        sample = (row["SAMPLE_ID"] or "").strip()
        subject = (row["PATIENT_ID"] or "").strip()
        cancer = (row["CANCER_TYPE"] or "").strip()
        if not sample:
            raise ValueError("Échantillon sans identifiant dans le fichier clinique.")
        if sample in sample_ids:
            raise ValueError("Échantillon en double dans le fichier clinique.")
        sample_ids.add(sample)
        if subject and cancer.casefold() not in UNKNOWN:
            labels[sample] = (subject, cancer)
            atlas.add_patient(cancer, subject)

    statuses = Counter()
    for row in _table(mutation_file, {"Tumor_Sample_Barcode", "Hugo_Symbol", "HGVSp_Short"}):
        sample = (row["Tumor_Sample_Barcode"] or "").strip()
        if sample not in sample_ids:
            raise ValueError("Échantillon mutationnel absent du fichier clinique : vérifiez que les deux fichiers ont la même version.")
        gene = (row["Hugo_Symbol"] or "").strip().upper()
        if gene not in panel or sample not in labels:
            continue
        status = (row.get("Mutation_Status") or "").strip().upper()
        if status not in SOMATIC | UNSPECIFIED:
            statuses["excluded_other_status"] += 1
            continue
        statuses["explicit_somatic" if status in SOMATIC else "unspecified"] += 1
        subject, cancer = labels[sample]
        atlas.add_variant(gene, row["HGVSp_Short"], cancer, subject)
    if not atlas.carriers:
        raise ValueError("Aucun variant protéique exploitable.")
    return atlas.export(
        output or cohort_path("genie"), source=f"AACR Project GENIE · {release}", source_url=GENIE,
        sequenced_samples=len(sample_ids), missing_labels=len(sample_ids) - len(labels),
        provenance=[{"study_id": release, "call_statuses": dict(statuses)}],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--mutations", required=True, type=Path)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    print(build_genie_atlas(args.clinical, args.mutations, args.release))


if __name__ == "__main__":
    main()
