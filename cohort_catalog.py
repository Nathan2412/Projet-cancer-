"""Sources distinctes pour explorer les observations sans additionner leurs effectifs."""

from pathlib import Path

from config import REPORTS_DIR

PORTAL = "https://www.cbioportal.org/study/summary?id="
TCGA = "https://www.cancer.gov/ccg/research/genome-sequencing/tcga"
GENIE = "https://aacrprojectgenie.org/data/"
MSK_LEGACY = {
    "msk_chord_2024": "MSK-CHORD",
    "msk_met_2021": "MSK MetTropism",
    "msk_impact_2017": "MSK-IMPACT 2017",
}
ICGC = "https://docs.icgc-argo.org/docs/data-access/icgc-25k-data"
ICGC_COHORTS = {
    "BLCA-CN": "Cancer de la vessie",
    "BOCA-FR": "Sarcome d’Ewing",
    "BOCA-UK": "Cancer des os",
    "BPLL-FR": "Leucémie prolymphocytaire B",
    "BRCA-KR": "Cancer du sein",
    "BTCA-JP": "Cancer des voies biliaires",
    "CLLE-ES": "Leucémie lymphoïde chronique",
    "ESCA-CN": "Cancer de l’œsophage",
    "GACA-JP": "Cancer de l’estomac",
    "LINC-JP": "Cancer du foie",
    "NACA-CN": "Cancer du nasopharynx",
    "NKTL-SG": "Lymphome NK/T",
    "ORCA-IN": "Cancer de la cavité buccale",
    "PAEN-AU": "Tumeur neuroendocrine du pancréas",
    "PAEN-IT": "Tumeur neuroendocrine du pancréas",
    "PRAD-CN": "Cancer de la prostate",
    "PRAD-FR": "Cancer de la prostate",
    "PRAD-UK": "Cancer de la prostate",
    "RECA-EU": "Carcinome rénal",
    "THCA-CN": "Cancer de la thyroïde",
    "THCA-SA": "Cancer de la thyroïde",
}
GDC_PROJECTS = {
    "CCDI-MCI": ("NCI · Initiative pédiatrique MCI", None),
    "CPTAC-3": ("NCI · CPTAC-3", None),
    "MP2PRT-ALL": ("NCI · MP2PRT leucémie lymphoblastique", "Leucémie aiguë lymphoblastique"),
    "ALCHEMIST-ALCH": ("NCI · ALCHEMIST poumon", "Cancer du poumon"),
    "MMRF-COMMPASS": ("NCI · CoMMpass myélome", "Myélome multiple"),
    "BEATAML1.0-COHORT": ("NCI · Beat AML", "Leucémie aiguë myéloïde"),
    "CPTAC-2": ("NCI · CPTAC-2", None),
    "CMI-MBC": ("NCI · Count Me In sein métastatique", "Cancer du sein métastatique"),
    "TARGET-OS": ("NCI · TARGET ostéosarcome", "Ostéosarcome"),
    "TARGET-ALL-P3": ("NCI · TARGET leucémie phase III", "Leucémie aiguë lymphoblastique"),
    "CGCI-HTMCP-CC": ("NCI · CGCI col de l’utérus", "Cancer du col de l’utérus"),
    "CMI-MPC": ("NCI · Count Me In prostate", "Cancer de la prostate métastatique"),
    "RC-PTCL": ("NCI · Lymphome T périphérique", "Lymphome T périphérique"),
    "CMI-ASC": ("NCI · Count Me In angiosarcome", "Angiosarcome"),
    "CGCI-BLGSP": ("NCI · CGCI lymphome de Burkitt", "Lymphome de Burkitt"),
    "EXCEPTIONAL_RESPONDERS-ER": ("NCI · Répondeurs exceptionnels", None),
}

COHORTS = {
    "origimed": ("OrigiMed · Chine", "pan_origimed_2020"),
    "tcga": ("TCGA PanCancer Atlas", None),
    "pcawg": ("PCAWG · ICGC / TCGA", "pancan_pcawg_2020"),
    "metabric": ("METABRIC · Sein", "brca_metabric"),
    "mds_iwg": ("IWG · Syndromes myélodysplasiques", "mds_iwg_2022"),
    "sysucc": ("SYSUCC · Côlon et rectum", "crc_sysucc_2022"),
    "cll_broad": ("Broad · Leucémie lymphoïde chronique", "cll_broad_2022"),
    "acc": ("Adenoid Cystic Carcinoma Project", "acc_2019"),
    "renal_sjuh": ("St. James’s · Rein", "ccrcc_sjuh_2023"),
    "ohsu": ("OHSU · Leucémie myéloïde aiguë", "aml_ohsu_2022"),
    "stjude": ("St. Jude · Leucémie pédiatrique", "aml_stjude_2024"),
    "dkfz": ("DKFZ · Tumeurs pédiatriques", "pediatric_dkfz_2017"),
    "mappyacts": ("MAPPYACTS · Tumeurs pédiatriques", "pancan_mappyacts_2022"),
    "target_nbl": ("TARGET · Neuroblastome", "nbl_target_2018_pub"),
    "target_wilms": ("TARGET · Tumeur de Wilms", "wt_target_2018_pub"),
    "target_bll": ("TARGET · Leucémie / lymphome B", "bll_target_gdc"),
    "target_aml": ("TARGET · Leucémie myéloïde aiguë", "aml_target_gdc"),
    "pog": ("POG · Tumeurs avancées", "pog570_bcgsc_2020"),
    "dfci_crc": ("Dana-Farber · Côlon et rectum", "coadread_dfci_2016"),
    "umich": ("Michigan · Tumeurs métastatiques", "metastatic_solid_tumors_mich_2017"),
    "dlbcl_duke": ("Duke · Lymphome diffus à grandes cellules B", "dlbcl_duke_2017"),
    "mbn_mdacc": ("MD Anderson · Tumeurs à cellules B", "mbn_mdacc_2013"),
    "glass": ("GLASS · Gliomes diffus", "difg_glass"),
    "cll_iuopa": ("IUOPA · Leucémie lymphoïde chronique", "cll_iuopa_2015"),
    "hcc_clca": ("CLCA · Foie", "hcc_clca_2024"),
    "prad_cpcg": ("CPC-GENE · Prostate", "prad_cpcg_2017"),
    "tracerx": ("TRACERx · Poumon", "nsclc_tracerx_2017"),
    "prad_su2c": ("SU2C / PCF · Prostate métastatique", "prad_su2c_2019"),
    "chol_icgc": ("ICGC · Cholangiocarcinome", "chol_icgc_2017"),
    "paad_qcmg": ("QCMG · Pancréas", "paad_qcmg_uq_2016"),
    "brca_mbc": ("MBCproject · Sein métastatique", "brca_mbcproject_2022"),
    "rcc_cptac": ("CPTAC · Rein", "rcc_cptac_gdc"),
    "prad_dkfz": ("DKFZ · Prostate", "prostate_dkfz_2018"),
    "luad_oncosg": ("OncoSG · Poumon", "luad_oncosg_2020"),
    "prad_p1000": ("P1000 · Prostate", "prad_p1000"),
    "mbn_sfu": ("Simon Fraser · Tumeurs à cellules B", "mbn_sfu_2023"),
    "bcc_unige": ("UNIGE · Carcinome basocellulaire", "bcc_unige_2016"),
    "coad_silu": ("Sidra / LUMC · Côlon", "coad_silu_2022"),
    "mixed_allen": ("Allen · Tumeurs solides MSS", "mixed_allen_2018"),
    "lung_nci": ("NCI · Poumon chez les non-fumeurs", "lung_nci_2022"),
    "brca_igr": ("Gustave Roussy · Sein métastatique", "brca_igr_2015"),
    "blca_bcan": ("BCAN / HCRN · Carcinome urothélial", "blca_bcan_hcrn_2022"),
    "thyroid_gatci": ("GATCI · Thyroïde anaplasique", "thyroid_gatci_2024"),
    "brca_smc": ("Samsung Medical Center · Sein", "brca_smc_2018"),
    "target_rhabdoid": ("TARGET · Tumeurs rhabdoïdes", "rt_target_2018_pub"),
    "msk_sarcoma": ("MSK · Sarcomes (chevauchement inconnu)", "sarcoma_msk_2022"),
}

OPTIONAL_COHORTS = {"brca_mbc", "target_rhabdoid"}
FIXED_LABELS = {
    "mds_iwg": "Myelodysplastic Syndromes",
    "cll_iuopa": "Chronic Lymphocytic Leukemia",
    "chol_icgc": "Cholangiocarcinoma",
}


def cohort_path(key):
    return Path(REPORTS_DIR) / f"allele_atlas_{key}.json"


def gdc_path(project):
    return cohort_path(f"gdc_{project.lower().replace('.', '_').replace('-', '_')}")


def icgc_path(project):
    return cohort_path(f"icgc_{project.lower().replace('-', '_')}")
