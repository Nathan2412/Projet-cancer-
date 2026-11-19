"""Importer les mutations somatiques ouvertes du NCI, sans compte ni identifiants persistés."""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from allele_atlas import AlleleAtlas
from cohort_catalog import GDC_PROJECTS, gdc_path
from import_public_atlas import PANEL, UNKNOWN

GDC = "https://api.gdc.cancer.gov"
SCOPE = "Cas avec MAF somatique ouvert ; annotations canoniques GDC, sauf l’équivalence BRAF V600E vérifiée sur GRCh38."
PORTAL = "https://portal.gdc.cancer.gov/projects/"
PANEL_SYMBOLS = set(PANEL.values())
PAGE_SIZE = 2000


def _query(endpoint, **params):
    data = urlencode({key: json.dumps(value) if key == "filters" else value for key, value in params.items()})
    request = Request(f"{GDC}/{endpoint}?{data}", headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=55) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == 2 or isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            sleep(2 ** attempt)


def _term(field, *values):
    return {"op": "in", "content": {"field": field, "value": list(values)}}


def _rows(endpoint, filters, fields, sort):
    offset = 0
    seen = set()
    while True:
        result = _query(endpoint, filters={"op": "and", "content": filters},
                        fields=fields, size=PAGE_SIZE, sort=sort, **{"from": offset})["data"]
        rows = result["hits"]
        for row in rows:
            identifier = row["id"]
            if identifier in seen:
                raise ValueError(f"Doublon dans la pagination GDC ({endpoint}).")
            seen.add(identifier)
            yield row
        offset += len(rows)
        if offset == result["pagination"]["total"]:
            return
        if not rows or offset > result["pagination"]["total"]:
            raise ValueError(f"Pagination GDC incomplète ({endpoint}).")


def case_label(case, fixed_label):
    if fixed_label:
        return fixed_label
    site = str(case.get("primary_site") or "").strip()
    diagnosis = str(case.get("disease_type") or "").strip()
    values = [value for value in (site, diagnosis) if value.casefold() not in UNKNOWN]
    return " · ".join(dict.fromkeys(values))


def protein_variants(row):
    ssm = row["ssm"]
    for consequence in ssm.get("consequence", []):
        transcript = consequence["transcript"]
        symbol = transcript.get("gene", {}).get("symbol")
        change = transcript.get("aa_change")
        if (transcript.get("is_canonical") is True and symbol in PANEL_SYMBOLS and change
                and "=" not in change):
            if (symbol == "BRAF" and change == "V640E"
                    and ssm.get("genomic_dna_change") == "chr7:g.140753336A>T"):
                change = "V600E"
            yield symbol, change


def collect_gdc_atlas(project, *, release):
    _, fixed_label = GDC_PROJECTS[project]
    maf_rows = _rows(
        "files",
        [_term("access", "open"), _term("data_type", "Masked Somatic Mutation"),
         _term("cases.project.project_id", project)],
        "access,cases.case_id,cases.primary_site,cases.disease_type,cases.project.project_id", "file_id:asc",
    )
    cases = {}
    profiled = set()
    open_files = 0
    for maf in maf_rows:
        if maf.get("access") != "open":
            raise ValueError("Le GDC a retourné un fichier hors accès ouvert.")
        open_files += 1
        for case in maf["cases"]:
            if case.get("project", {}).get("project_id") != project:
                continue
            subject = case["case_id"]
            profiled.add(subject)
            label = case_label(case, fixed_label)
            if label:
                if subject in cases and cases[subject] != label:
                    raise ValueError("Types tumoraux contradictoires pour un cas.")
                cases[subject] = label
    if not cases:
        raise ValueError("Aucun cas étiqueté lié à un fichier MAF ouvert.")

    atlas = AlleleAtlas()
    for subject, cancer in cases.items():
        atlas.add_patient(cancer, subject)
    occurrences = _rows(
        "ssm_occurrences",
        [_term("case.project.project_id", project), _term("ssm.consequence.transcript.gene.symbol", *sorted(PANEL_SYMBOLS))],
        "case.case_id,case.project.project_id,ssm.genomic_dna_change,ssm.consequence.transcript.gene.symbol,"
        "ssm.consequence.transcript.is_canonical,ssm.consequence.transcript.aa_change", "ssm_occurrence_id:asc",
    )
    stats = Counter()
    for occurrence in occurrences:
        case = occurrence["case"]
        subject = case["case_id"]
        if case["project"]["project_id"] != project:
            raise ValueError("Occurrence hors du projet demandé.")
        if subject not in cases:
            stats["outside_open_labelled_cases"] += 1
            continue
        for symbol, change in protein_variants(occurrence):
            atlas.add_variant(symbol, change, cases[subject], subject)
            stats["canonical_somatic_annotations"] += 1
    if not atlas.carriers:
        raise ValueError("Aucun variant protéique canonique dans les cas ouverts.")
    return atlas, {"study_id": project, "release": release, "open_maf_files": open_files,
                   "cases_with_open_maf": len(profiled), "call_statuses": dict(stats), "scope": SCOPE}


def build_gdc_atlas(project, *, release, path=None):
    atlas, metadata = collect_gdc_atlas(project, release=release)
    patients = len(set().union(*atlas.groups.values()))
    result = atlas.export(
        path or gdc_path(project), source=GDC_PROJECTS[project][0], source_url=f"{PORTAL}{project}",
        missing_labels=metadata["cases_with_open_maf"] - patients, provenance=[metadata],
    )
    return project, patients, len(atlas.carriers), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", metavar="PROJET", nargs="+")
    parser.add_argument("--jobs", choices=range(1, 5), type=int, default=3)
    args = parser.parse_args()
    projects = list(GDC_PROJECTS) if args.projects == ["all"] else list(dict.fromkeys(args.projects))
    if unknown := set(projects) - GDC_PROJECTS.keys():
        parser.error(f"Projets non pris en charge : {', '.join(sorted(unknown))}.")
    release = _query("status")["data_release"]

    def run(project):
        try:
            key, patients, variants, path = build_gdc_atlas(project, release=release)
            print(f"{key} : {patients} patients · {variants} variants · {path}", flush=True)
            return True
        except (URLError, OSError, ValueError, KeyError) as exc:
            print(f"ÉCHEC {project} : {exc}", flush=True)
            return False

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        done = list(pool.map(run, projects))
    return not all(done)


if __name__ == "__main__":
    raise SystemExit(main())
