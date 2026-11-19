"""Index agrégé des changements protéiques observés dans des cohortes tumorales."""

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

from config import REPORTS_DIR

ATLAS_PATHS = {
    "MSK-IMPACT 50K": Path(REPORTS_DIR) / "allele_atlas_msk50k.json",
    "Cohortes locales": Path(REPORTS_DIR) / "allele_atlas_local.json",
}
MSK_STUDY_URL = "https://www.cbioportal.org/study/summary?id=msk_impact_50k_2026"
SCHEMA_VERSION = 1
_AA3 = dict(zip(
    ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "TER"),
    "ARNDCQEGHILKMFPSTWYV*",
    strict=True,
))


def normalize_protein_change(value):
    token = str(value or "").strip().upper().removeprefix("P.").strip("()")
    match = re.fullmatch(r"([A-Z]{3})([0-9]+)([A-Z]{3})", token)
    if match and match[1] in _AA3 and match[3] in _AA3:
        return f"{_AA3[match[1]]}{match[2]}{_AA3[match[3]]}"
    if re.fullmatch(r"[A-Z*][0-9]+[A-Z_*?=+\-][A-Z0-9_*?=+\-]*", token):
        return token
    return ""


class AlleleAtlas:
    def __init__(self):
        self.groups = defaultdict(set)
        self.carriers = defaultdict(lambda: defaultdict(set))

    def add_patient(self, cancer, subject):
        self.groups[cancer].add(subject)

    def add_variant(self, gene, protein_change, cancer, subject):
        protein = normalize_protein_change(protein_change)
        if gene and protein:
            self.carriers[f"{gene.upper()} {protein}"][cancer].add(subject)

    def payload(self, *, source, source_url=None, sequenced_samples=None, missing_labels=0,
                is_limited=False, provenance=None):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "source_url": source_url,
            "subjects": len(set().union(*self.groups.values())),
            "sequenced_samples": sequenced_samples,
            "missing_labels": missing_labels,
            "is_limited": is_limited,
            "cancers": {cancer: len(subjects) for cancer, subjects in sorted(self.groups.items())},
            "alleles": {
                name: {
                    "subjects": len(set().union(*groups.values())),
                    "cancers": {cancer: len(subjects) for cancer, subjects in sorted(groups.items())},
                }
                for name, groups in sorted(self.carriers.items())
            },
        }
        if provenance is not None:
            payload["provenance"] = provenance
        return payload

    def export(self, path, **kwargs):
        return write_atlas(path, self.payload(**kwargs))


def write_atlas(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def build_local_atlas(rows, *, is_limited=False):
    from clinical_rules import normalize_cancer_label

    atlas = AlleleAtlas()
    missing_labels = 0
    for row in rows:
        metadata = row["metadata"]
        cancer = normalize_cancer_label(metadata.get("cancer_type"))
        if not cancer:
            missing_labels += 1
            continue
        original = str(metadata.get("original_patient_id") or "").strip().upper()
        subject = ("source", original) if original else ("local", row["patient_id"])
        atlas.add_patient(cancer, subject)
        mutations = row.get("mutations")
        if mutations is None:
            mutations = (mut for gene in row["gene_analyses"].values() for mut in gene["mutations"])
        for mut in mutations:
            atlas.add_variant(mut.get("gene"), mut.get("protein_change"), cancer, subject)
    return atlas.export(
        ATLAS_PATHS["Cohortes locales"], source="Cohortes locales cBioPortal (TCGA et autres études)",
        source_url="https://www.cbioportal.org/", missing_labels=missing_labels, is_limited=is_limited,
    )


def matching_names(data, query):
    query = query.strip().upper()
    names = data["alleles"]
    if not query:
        return sorted(names, key=lambda name: (-names[name]["subjects"], name))
    tokens = query.split()
    if len(tokens) > 1:
        protein = normalize_protein_change("".join(tokens[1:]))
        exact = f"{tokens[0]} {protein}"
        if protein:
            return [exact] if exact in names else []
        variant_query = "".join(tokens[1:]).removeprefix("P.")
        return [name for name in names if name.startswith(f"{tokens[0]} {variant_query}")]
    protein = normalize_protein_change(query)
    if protein:
        return [name for name in names if name.endswith(f" {protein}")]
    return [name for name in names if name.startswith(query) or name.partition(" ")[2].startswith(query)]


def main():
    from loader import get_patient_list_real, load_patient_data_real

    parser = argparse.ArgumentParser(description="Créer l’index agrégé allèle → types tumoraux.")
    parser.add_argument("--source", choices=["local", "msk50k"], default="local")
    parser.add_argument("--extend", action="store_true", help="Mettre à jour uniquement les 24 gènes ajoutés à l’index MSK existant.")
    args = parser.parse_args()
    if args.extend and args.source != "msk50k":
        parser.error("--extend nécessite --source msk50k.")
    if args.source == "msk50k":
        from import_msk_atlas import build_msk_atlas

        print(build_msk_atlas(extend=args.extend))
        return
    patients = get_patient_list_real()
    if not patients:
        parser.error("Aucun dossier réel local. Utilisez --source msk50k ou chargez les cohortes locales.")
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = (patients[start:start + 256] for start in range(0, len(patients), 256))
        rows = (row for batch in batches for row in pool.map(load_patient_data_real, batch))
        print(build_local_atlas(rows))


if __name__ == "__main__":
    main()
