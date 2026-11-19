"""Lire les fragments somatiques ouverts de l’archive ICGC 25K, sans compte."""

import argparse
import gzip
import io
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests

from allele_atlas import AlleleAtlas
from cohort_catalog import ICGC, ICGC_COHORTS, icgc_path
from import_gdc_atlas import PANEL_SYMBOLS, _rows, _term

BUCKET = "https://object.genomeinformatics.org/icgc25k-open"
RELEASE = "release_28"
STATE = threading.local()
CONSEQUENCES = {"missense_variant", "stop_gained", "stop_lost"}


def _session():
    if not hasattr(STATE, "session"):
        STATE.session = requests.Session()
        STATE.session.mount("https://", requests.adapters.HTTPAdapter(max_retries=2))
    return STATE.session


def _read_object(url, parse):
    for attempt in range(3):
        try:
            with _session().get(url, stream=True, timeout=(15, 40)) as response:
                response.raise_for_status()
                return parse(response.raw)
        except (requests.RequestException, OSError, EOFError) as exc:
            if attempt == 2 or isinstance(exc, requests.HTTPError) and 400 <= exc.response.status_code < 429:
                raise
            sleep(2 ** attempt)


def list_mutation_objects(project):
    prefix = f"{RELEASE}/data/{project}/"
    params = {"list-type": 2, "prefix": prefix, "max-keys": 1000}
    seen = set()
    while True:
        root = _read_object(f"{BUCKET}?{urlencode(params)}", lambda handle: ElementTree.parse(handle).getroot())
        for item in root.findall("{*}Contents"):
            key = item.findtext("{*}Key")
            match = re.fullmatch(re.escape(prefix) + r"(DO[0-9]+)/ssm_open/part-[0-9]+\.gz", key)
            if match:
                if key in seen:
                    raise ValueError("Clé d’objet ICGC en double.")
                seen.add(key)
                yield key, match[1]
        if root.findtext("{*}IsTruncated") != "true":
            return
        token = root.findtext("{*}NextContinuationToken")
        if not token or token == params.get("continuation-token"):
            raise ValueError("Pagination ICGC bloquée.")
        params["continuation-token"] = token


def _header():
    url = f"{BUCKET}/{RELEASE}/headers/ssm_open.tsv.gz"
    return _read_object(url, lambda handle: gzip.GzipFile(fileobj=handle).read().decode().strip().split("\t"))


def _transcripts():
    genes = _rows("genes", [_term("symbol", *sorted(PANEL_SYMBOLS))],
                  "gene_id,symbol,canonical_transcript_id", "gene_id:asc")
    return {
        row["gene_id"]: (row["symbol"], row["canonical_transcript_id"].split(".")[0])
        for row in genes if row.get("canonical_transcript_id") and row["symbol"] in PANEL_SYMBOLS
    }


def parse_fragment(handle, columns, project, donor, transcripts):
    variants = set()
    has_rows = False
    subject_idx, project_idx, consequence_idx, protein_idx, gene_idx, transcript_idx = columns
    with gzip.GzipFile(fileobj=handle) as zipped, io.TextIOWrapper(zipped, encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            row = line.rstrip("\r\n").split("\t", transcript_idx + 1)
            if len(row) <= transcript_idx or row[subject_idx] != donor or row[project_idx] != project:
                raise ValueError("Ligne ICGC hors du donneur ou du projet demandé.")
            has_rows = True
            if row[consequence_idx] not in CONSEQUENCES:
                continue
            gene = transcripts.get(row[gene_idx].split(".")[0])
            if gene and row[transcript_idx].split(".")[0] == gene[1]:
                protein = row[protein_idx].strip()
                if re.fullmatch(r"[A-Z][1-9][0-9]*[A-Z*]", protein) and protein[0] != protein[-1]:
                    variants.add((gene[0], protein))
    return donor, has_rows, variants


def collect_icgc_atlas(project, objects, columns, transcripts, pool):
    label = ICGC_COHORTS[project]
    atlas = AlleleAtlas()
    counts = Counter()

    def fetch(item):
        key, donor = item
        return _read_object(f"{BUCKET}/{key}", lambda handle: parse_fragment(handle, columns, project, donor, transcripts))

    for donor, observed, variants in pool.map(fetch, objects):
        counts["open_fragments"] += 1
        if observed:
            atlas.add_patient(label, donor)
        for symbol, protein in variants:
            atlas.add_variant(symbol, protein, label, donor)
    if not atlas.carriers:
        raise ValueError("Aucune substitution canonique dans le panel.")
    return atlas, {"study_id": project, "release": RELEASE, "call_statuses": dict(counts),
                   "scope": "Donneurs avec appels somatiques ouverts ; substitutions protéiques canoniques uniquement."}


def build_icgc_atlas(project, objects, columns, transcripts, pool):
    atlas, metadata = collect_icgc_atlas(project, objects, columns, transcripts, pool)
    result = atlas.export(
        icgc_path(project), source=f"ICGC · {project}", source_url=ICGC,
        provenance=[metadata],
    )
    return len(atlas.groups[ICGC_COHORTS[project]]), len(atlas.carriers), result


def mutation_columns():
    header = _header()
    return tuple(header.index(field) for field in (
        "icgc_donor_id", "project_code", "consequence_type", "aa_mutation", "gene_affected", "transcript_affected",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", metavar="PROJET", nargs="+")
    parser.add_argument("--jobs", choices=range(1, 9), type=int, default=6)
    args = parser.parse_args()
    projects = list(ICGC_COHORTS) if args.projects == ["all"] else list(dict.fromkeys(args.projects))
    if unknown := set(projects) - ICGC_COHORTS.keys():
        parser.error(f"Projets inconnus : {', '.join(sorted(unknown))}.")
    columns = mutation_columns()
    transcripts = _transcripts()
    if not transcripts:
        raise ValueError("Aucun transcript canonique disponible.")
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for project in projects:
            try:
                objects = list(list_mutation_objects(project))
                print(f"{project} : {len(objects)} fragments à lire…", flush=True)
                subjects, variants, path = build_icgc_atlas(project, objects, columns, transcripts, pool)
                print(f"{project} : {subjects} donneurs · {variants} variants · {path}", flush=True)
            except (requests.RequestException, OSError, ValueError, KeyError) as exc:
                failures.append(project)
                print(f"ÉCHEC {project} : {exc}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
