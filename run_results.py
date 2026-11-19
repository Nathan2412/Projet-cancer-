"""Publication d'un instantané agrégé du run, sans identifiants patients."""

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from config import REPORTS_DIR

SUMMARY_PATH = Path(REPORTS_DIR) / "run_summary.json"
SCHEMA_VERSION = 1


def publish_run_summary(results, genes, matrix, evaluation, *, preparation_seconds,
                        elapsed_seconds, is_limited):
    labels = Counter(row["metadata"].get("cancer_type") or "Non renseigné" for row in results)
    studies = Counter(
        row["metadata"].get("study_id") or row["metadata"].get("original_study") or "Non renseignée"
        for row in results
    )
    gene_totals = Counter()
    gene_patients = Counter()
    for counts in matrix.values():
        gene_totals.update(counts)
        gene_patients.update(gene for gene, count in counts.items() if count > 0)
    genes_summary = [
        {"gene": gene, "patients": gene_patients[gene], "variants": gene_totals[gene]}
        for gene in genes
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "cBioPortal · cohortes TCGA, MSK-IMPACT et études complémentaires",
        "is_limited": is_limited,
        "patients": len(results),
        "variants": sum(row["total_mutations_detected"] for row in results),
        "preparation_seconds": round(preparation_seconds, 2),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "cancers": dict(labels.most_common()),
        "studies": dict(studies.most_common()),
        "genes": sorted(genes_summary, key=lambda row: (-row["patients"], row["gene"])),
        "evaluation": evaluation,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=SUMMARY_PATH.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp_path, SUMMARY_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return SUMMARY_PATH
