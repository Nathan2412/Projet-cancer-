"""Import de la cohorte MSK-IMPACT 50K dans un index local, sans données individuelles persistées."""

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from allele_atlas import ATLAS_PATHS, MSK_STUDY_URL, SCHEMA_VERSION, AlleleAtlas, write_atlas
from download_real_data import CBIOPORTAL_API, ENTREZ_TO_GENE

STUDY = "msk_impact_50k_2026"
PAGE_SIZE = 10_000
BATCH_SIZE = 750
EXTRA_GENES = {
    4893: "NRAS", 3265: "HRAS", 3815: "KIT", 5156: "PDGFRA",
    4233: "MET", 2260: "FGFR1", 2263: "FGFR2", 2261: "FGFR3",
    207: "AKT1", 5604: "MAP2K1", 1499: "CTNNB1", 6597: "SMARCA4",
    472: "ATM", 11200: "CHEK2", 79728: "PALB2", 5426: "POLE",
    2099: "ESR1", 9817: "KEAP1", 4780: "NFE2L2", 2778: "GNAS",
    7248: "TSC1", 7249: "TSC2", 3717: "JAK2", 2065: "ERBB3",
}


def _api(path, *, params=None, body=None):
    url = f"{CBIOPORTAL_API}/{path}"
    if params:
        url += f"?{urlencode(params)}"
    request = Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        headers={"Accept": "application/json", "Content-Type": "application/json", "Accept-Encoding": "gzip"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                if response.headers.get("Content-Encoding") == "gzip":
                    with gzip.GzipFile(fileobj=response) as content:
                        return json.load(content)
                return json.load(response)
        except HTTPError as exc:
            if attempt == 3 or exc.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdecimal() else 2 ** attempt
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            delay = 2 ** attempt
        sleep(delay)


def _sample_labels(sample_ids):
    labels = {}
    page = 0
    while True:
        rows = _api(f"studies/{STUDY}/clinical-data", params={
            "clinicalDataType": "SAMPLE", "attributeId": "CANCER_TYPE",
            "pageSize": PAGE_SIZE, "pageNumber": page,
        })
        for row in rows:
            if row["clinicalAttributeId"] != "CANCER_TYPE":
                raise ValueError("L’API n’a pas appliqué le filtre CANCER_TYPE.")
            if row["sampleId"] not in sample_ids:
                continue
            if row["sampleId"] in labels:
                raise ValueError("Échantillon en double dans les pages cliniques cBioPortal.")
            label = row["value"].strip()
            labels[row["sampleId"]] = (row["patientId"], label)
        if len(rows) < PAGE_SIZE:
            break
        page += 1
    return {
        sample: (subject, cancer) for sample, (subject, cancer) in labels.items()
        if subject and cancer and cancer.lower() not in {"unknown", "not reported", "na", "n/a", "[not available]"}
    }


def _fetch_batch(sample_batch, gene_ids):
    rows = _api(f"molecular-profiles/{STUDY}_mutations/mutations/fetch", params={
        "projection": "SUMMARY", "pageSize": PAGE_SIZE, "pageNumber": 0,
    }, body={"sampleIds": sample_batch, "entrezGeneIds": gene_ids})
    if len(rows) >= PAGE_SIZE:
        pivot = len(sample_batch) // 2
        if not pivot:
            raise ValueError("Réponse tronquée pour un échantillon cBioPortal.")
        return _fetch_batch(sample_batch[:pivot], gene_ids) + _fetch_batch(sample_batch[pivot:], gene_ids)
    return rows


def merge_panel(existing, addition):
    cohort_keys = ("schema_version", "source", "source_url", "subjects", "sequenced_samples", "missing_labels", "cancers")
    if any(existing[key] != addition[key] for key in cohort_keys):
        raise ValueError("Les cohortes ne correspondent pas ; reconstruisez le catalogue MSK en entier.")
    extra_symbols = set(EXTRA_GENES.values())
    retained = {name: value for name, value in existing["alleles"].items() if name.split(" ", 1)[0] not in extra_symbols}
    return dict(addition, alleles=retained | addition["alleles"])


def build_msk_atlas(*, extend=False):
    target = ATLAS_PATHS["MSK-IMPACT 50K"]
    existing = json.loads(target.read_text(encoding="utf-8")) if extend else None
    if existing is not None and (existing.get("schema_version") != SCHEMA_VERSION or existing.get("source_url") != MSK_STUDY_URL):
        raise ValueError("Index MSK incompatible ; reconstruisez le catalogue sans --extend.")
    panel = EXTRA_GENES if extend else ENTREZ_TO_GENE | EXTRA_GENES
    gene_ids = tuple(panel)
    study = _api(f"studies/{STUDY}")
    sample_ids = _api(f"sample-lists/{STUDY}_sequenced/sample-ids")
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or len(sample_ids) != study["sequencedSampleCount"]:
        raise ValueError("La liste cBioPortal ne correspond pas à l’effectif séquencé annoncé.")
    labels = _sample_labels(set(sample_ids))
    if not labels:
        raise ValueError("Aucun échantillon clinique étiqueté pour MSK-IMPACT 50K.")
    print(f"MSK-IMPACT 50K : {len(labels):,} échantillons étiquetés sur {len(sample_ids):,} ; {len(panel)} gènes demandés.", flush=True)
    atlas = AlleleAtlas()
    for subject, cancer in labels.values():
        atlas.add_patient(cancer, subject)

    labelled_samples = sorted(labels)
    batches = [labelled_samples[start:start + BATCH_SIZE] for start in range(0, len(labels), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (batch, mutations) in enumerate(zip(
            batches, pool.map(lambda batch: _fetch_batch(batch, gene_ids), batches), strict=True,
        ), 1):
            requested = set(batch)
            for row in mutations:
                sample = row["sampleId"]
                if sample not in requested:
                    raise ValueError("L’API a retourné une mutation hors du lot demandé.")
                if str(row.get("mutationStatus") or "").upper() not in {"SOMATIC", "LIKELY SOMATIC"}:
                    continue
                subject, cancer = labels[sample]
                atlas.add_variant(panel.get(row["entrezGeneId"]), row.get("proteinChange"), cancer, subject)
            if index == len(batches) or index % 10 == 0:
                print(f"Mutations : {index}/{len(batches)} lots.", flush=True)

    if not atlas.carriers:
        raise ValueError("Aucune mutation somatique annotée dans le panel reçu.")
    payload = atlas.payload(
        source=study["name"], source_url=MSK_STUDY_URL,
        sequenced_samples=len(sample_ids), missing_labels=len(sample_ids) - len(labels),
    )
    if existing is not None:
        payload = merge_panel(existing, payload)
    return write_atlas(target, payload)
