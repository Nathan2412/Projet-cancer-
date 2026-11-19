"""Tableau de bord local des résultats agrégés du pipeline."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from allele_atlas import ATLAS_PATHS, SCHEMA_VERSION as ATLAS_SCHEMA_VERSION, matching_names
from cohort_catalog import COHORTS, GDC_PROJECTS, GENIE, ICGC_COHORTS, cohort_path, gdc_path, icgc_path
from config import REPORTS_DIR
from run_results import SCHEMA_VERSION, SUMMARY_PATH

TEAL = "#007D70"
INDIGO = "#6552AC"
PAGE_SIZE = 40
TRANSFER_PATH = Path(REPORTS_DIR) / "external_cohort_validation.json"
MULTICOHORT_PATH = Path(REPORTS_DIR) / "multicohort_validation.json"
SHORTCUTS = (
    "BRAF V600E", "KRAS G12D", "EGFR L858R", "PIK3CA H1047R", "NRAS Q61R", "AKT1 E17K",
    "TP53 R175H", "IDH1 R132H", "ESR1 Y537S", "FGFR3 S249C", "CTNNB1 S45F", "KIT D816V",
)
FRENCH_CANCERS = {
    "Bladder Cancer": "Cancer de la vessie",
    "Bone Cancer": "Cancer des os",
    "Breast Cancer": "Cancer du sein",
    "Cancer of Unknown Primary": "Cancer de primitif inconnu",
    "Cervical Cancer": "Cancer du col de l’utérus",
    "Colorectal Cancer": "Cancer colorectal",
    "Colorectal Carcinoma": "Cancer colorectal",
    "Breast Carcinoma": "Cancer du sein",
    "Endometrial Cancer": "Cancer de l’endomètre",
    "Esophagogastric Cancer": "Cancer œsogastrique",
    "Gastrointestinal Stromal Tumor": "Tumeur stromale gastro-intestinale",
    "Germ Cell Tumor": "Tumeur germinale",
    "Glioma": "Gliome",
    "Head and Neck Cancer": "Cancer de la tête et du cou",
    "Hepatobiliary Cancer": "Cancer hépatobiliaire",
    "Lung Cancer": "Cancer du poumon",
    "Melanoma": "Mélanome",
    "Mesothelioma": "Mésothéliome",
    "Non-Small Cell Lung Cancer": "Cancer du poumon non à petites cellules",
    "Non Small Cell Lung Cancer": "Cancer du poumon non à petites cellules",
    "Ovarian Cancer": "Cancer de l’ovaire",
    "Pancreatic Cancer": "Cancer du pancréas",
    "Prostate Cancer": "Cancer de la prostate",
    "Renal Cell Carcinoma": "Carcinome rénal",
    "Small Cell Lung Cancer": "Cancer du poumon à petites cellules",
    "Soft Tissue Sarcoma": "Sarcome des tissus mous",
    "Thyroid Cancer": "Cancer de la thyroïde",
    "Thyroid Carcinoma": "Cancer de la thyroïde",
    "Uterine Sarcoma": "Sarcome de l’utérus",
    "Melanome": "Mélanome",
    "Thyroide": "Thyroïde",
    "Uterus": "Utérus",
    "TeteEtCou": "Cancer de la tête et du cou",
}


@st.cache_data(max_entries=4)
def load_summary(path, revision):
    with open(path, encoding="utf-8") as handle:
        run = json.load(handle)
    if run.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Version de rapport non prise en charge. Relancez l’analyse.")
    return run


@st.cache_data(max_entries=4)
def load_atlas(path, revision):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != ATLAS_SCHEMA_VERSION:
        raise ValueError("Format d’index non reconnu ; reconstruisez l’index.")
    alleles = data["alleles"]
    ranks = sorted(alleles, key=lambda name: (-alleles[name]["subjects"], name))
    return data, ranks, sorted({name.split(" ", 1)[0] for name in ranks})


@st.cache_data(max_entries=1)
def load_comparison_index(file_revisions):
    variants = {}
    sources = {}
    for source, path, _revision in file_revisions:
        try:
            with open(path, encoding="utf-8") as handle:
                atlas = json.load(handle)
            if atlas.get("schema_version") != ATLAS_SCHEMA_VERSION:
                raise ValueError("Format inconnu")
            sources[source] = {
                "patients": atlas["subjects"], "url": atlas["source_url"],
                "variants": len(atlas["alleles"]), "types": len(atlas["cancers"]),
                "samples": atlas["sequenced_samples"], "catalogues": len(atlas.get("provenance", [])),
            }
            for variant, record in atlas["alleles"].items():
                labels = record["cancers"]
                largest = max(labels.values())
                lead = tuple(label for label, value in sorted(labels.items()) if value == largest)
                variants.setdefault(variant, {})[source] = (record["subjects"], lead, largest)
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"Catalogue illisible ({source}) : {exc}") from exc
    ranks = sorted(variants, key=lambda name: (-len(variants[name]), -max(v[0] for v in variants[name].values()), name))
    return variants, sources, ranks


def source_files():
    paths = [(label, path) for label, path in ATLAS_PATHS.items() if label != "Cohortes locales"]
    paths.append(("MSK · Complément hors 50K", cohort_path("msk_legacy_unique")))
    paths += [(label, gdc_path(project)) for project, (label, _) in GDC_PROJECTS.items()]
    paths += [(label, cohort_path(key)) for key, (label, _) in COHORTS.items()]
    paths += [(f"ICGC · {project} ({cancer})", icgc_path(project)) for project, cancer in ICGC_COHORTS.items()]
    paths += [("AACR Project GENIE", cohort_path("genie")), ("Cohortes locales", ATLAS_PATHS["Cohortes locales"])]
    files = []
    for label, path in paths:
        try:
            files.append((label, str(path), path.stat().st_mtime_ns))
        except FileNotFoundError:
            continue
    return tuple(files)


def count(value):
    return f"{value:,}".replace(",", " ")


def percent(value):
    return f"{value:.1%}".replace(".", ",")


def display_cancer(label):
    return FRENCH_CANCERS.get(label, label)


def select_shortcut(name):
    st.session_state.active_variant = name
    st.session_state.variant_search = ""
    st.session_state.variant_gene = None


def select_from_picker():
    st.session_state.active_variant = st.session_state.variant_picker


def render_variant_bar(atlas, ranked_names, genes):
    alleles = atlas["alleles"]
    with st.container(key="variant_toolbar", border=True):
        search_col, gene_col = st.columns([3, 1], vertical_alignment="bottom")
        query = search_col.text_input(
            "Rechercher un allèle", key="variant_search", max_chars=100,
            placeholder="Ex. BRAF V600E, KRAS ou R175H",
            help="Tapez un gène, une position ou une notation protéique (ex. BRAF p.Val600Glu), puis appuyez sur Entrée.",
        )
        if st.session_state.get("variant_gene") not in genes:
            st.session_state.variant_gene = None
        gene = gene_col.selectbox("Filtrer par gène", genes, index=None, key="variant_gene", placeholder="Tous les gènes")
        candidates = set(matching_names(atlas, query)) if query.strip() else None
        gene_prefix = f"{gene} " if gene else ""
        matches = [
            name for name in ranked_names if name.startswith(gene_prefix) and (candidates is None or name in candidates)
        ]
        if not matches:
            st.info("Aucun résultat dans cette cohorte pour ces filtres.")
            st.button("Effacer les filtres", on_click=select_shortcut, args=(None,))
            return None

        selected = st.session_state.get("active_variant")
        if selected is None:
            selected = next((name for name in SHORTCUTS if name in alleles), None)
        if selected not in matches:
            selected = matches[0]
        st.session_state.active_variant = selected
        position = matches.index(selected)
        start = position // PAGE_SIZE * PAGE_SIZE
        previous_col, picker_col, next_col = st.columns([1, 4, 1], vertical_alignment="bottom")
        previous_col.button("← Précédent", disabled=position == 0, key="previous_variant", width="stretch",
                            on_click=st.session_state.update, args=({"active_variant": matches[max(0, position - 1)]},))
        if st.session_state.get("variant_picker") != selected:
            st.session_state.variant_picker = selected
        picker_col.selectbox(
            "Variant affiché", matches[start:start + PAGE_SIZE], key="variant_picker", index=None,
            on_change=select_from_picker, help="Ouvrez la liste ou utilisez les flèches pour parcourir les résultats.",
        )
        next_col.button("Suivant →", disabled=position == len(matches) - 1, key="next_variant", width="stretch",
                        on_click=st.session_state.update, args=({"active_variant": matches[min(len(matches) - 1, position + 1)]},))

        page_count = (len(matches) + PAGE_SIZE - 1) // PAGE_SIZE
        caption_col, pages_col = st.columns([2, 1], vertical_alignment="center")
        caption_col.caption(f"Variant {count(position + 1)} sur {count(len(matches))} · tri par nombre de patients")
        if page_count > 1:
            page = start // PAGE_SIZE + 1
            page_key = f"variant_page_{page}_{page_count}"
            pages_col.number_input(
                "Page du catalogue", min_value=1, max_value=page_count, step=1, value=page,
                key=page_key, help=f"{count(page_count)} pages de {PAGE_SIZE} variants maximum.",
                on_change=jump_to_page, args=(matches, page_key),
            )
    return selected


def jump_to_page(matches, widget_key, selection_key="active_variant"):
    st.session_state[selection_key] = matches[(st.session_state[widget_key] - 1) * PAGE_SIZE]


def render_allele_lookup():
    files = source_files()
    available = {label: (path, revision) for label, path, revision in files}
    if not available:
        st.info("Aucun index d’allèles disponible pour le moment.")
        st.code("python allele_atlas.py --source msk50k", language="powershell")
        return
    heading, source_col = st.columns([3, 1], vertical_alignment="bottom")
    heading.markdown("### Trouver un variant")
    compare = "Comparer les cohortes"
    source = source_col.selectbox("Cohorte de référence", [compare, *available] if len(available) > 1 else list(available),
                                  key="atlas_source")
    if source == compare:
        render_comparison(files)
        return
    path, revision = available[source]
    try:
        atlas, ranked_names, genes = load_atlas(path, revision)
    except (OSError, ValueError) as exc:
        st.error(f"Impossible de lire l’index d’allèles : {exc}")
        return

    st.caption(
        f"**{count(len(ranked_names))} variants** · {len(genes)} gènes · "
        f"{count(atlas['subjects'])} patients distincts · {len(atlas['cancers'])} types tumoraux"
    )
    if atlas["source_url"]:
        st.caption(f"[Consulter la source]({atlas['source_url']})")
    if not ranked_names:
        st.info("Cette cohorte ne contient aucun variant indexé.")
        return
    if atlas["is_limited"]:
        st.warning("Cette source ne contient qu’un sous-ensemble technique des dossiers locaux.")
    if any(row["call_statuses"].get("unspecified") for row in atlas.get("provenance", [])):
        st.caption("Le profil tumoral inclut des appels dont le statut somatique n’est pas renseigné par la source.")
    for note in dict.fromkeys(row["scope"] for row in atlas.get("provenance", []) if row.get("scope")):
        st.caption(note)

    selected = render_variant_bar(atlas, ranked_names, genes)
    if selected is not None:
        render_allele_summary(atlas, selected)
    shortcuts = [name for name in SHORTCUTS if name in atlas["alleles"]]
    if shortcuts:
        st.caption("Accès rapide")
        with st.container(horizontal=True, key="variant_shortcuts"):
            for name in shortcuts:
                st.button(name, key=f"shortcut_{name}", on_click=select_shortcut, args=(name,))
    if selected is not None:
        render_tumor_breakdown(atlas, selected)


def render_comparison(files):
    try:
        variants, sources, ranks = load_comparison_index(files)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(f"**{count(len(variants))} variants distincts** · {len(sources)} catalogues · effectifs de patients non additionnés")
    st.write("**Tapez un variant ou choisissez un raccourci** pour comparer les observations entre cohortes.")
    query = st.text_input("Rechercher dans toutes les cohortes", key="compare_query",
                          placeholder="Ex. BRAF V600E, KRAS ou ESR1 Y537S", max_chars=100)
    shortcuts = [name for name in SHORTCUTS[:6] if name in variants]
    with st.container(horizontal=True, key="variant_shortcuts"):
        for name in shortcuts:
            st.button(name, key=f"compare_{name}", on_click=st.session_state.update, args=({"compare_query": name},))
    candidates = set(matching_names({"alleles": variants}, query)) if query.strip() else None
    if not candidates and query.strip():
        st.info("Aucun variant correspondant dans les catalogues chargés.")
        return
    matches = ranks if candidates is None else [name for name in ranks if name in candidates]
    if not matches:
        st.info("Aucun variant indexé.")
        return
    selected = st.session_state.get("compare_active", SHORTCUTS[0])
    if selected not in matches:
        selected = matches[0]
    st.session_state.compare_active = selected
    position = matches.index(selected)
    offset = position // PAGE_SIZE * PAGE_SIZE
    prev_col, select_col, next_col = st.columns([1, 4, 1], vertical_alignment="bottom")
    prev_col.button(
        "← Précédent", disabled=position == 0, width="stretch", key="compare_previous",
        on_click=st.session_state.update, args=({"compare_active": matches[max(0, position - 1)]},),
    )
    if st.session_state.get("compare_variant") != selected:
        st.session_state.compare_variant = selected
    select_col.selectbox(
        "Variant à comparer", matches[offset:offset + PAGE_SIZE], index=None, key="compare_variant",
        on_change=pick_comparison,
    )
    next_col.button(
        "Suivant →", disabled=position == len(matches) - 1, width="stretch", key="compare_next",
        on_click=st.session_state.update, args=({"compare_active": matches[min(len(matches) - 1, position + 1)]},),
    )
    position_col, page_col = st.columns([2, 1], vertical_alignment="center")
    position_col.caption(f"Variant {count(position + 1)} sur {count(len(matches))} · tri par nombre de catalogues")
    if len(matches) > PAGE_SIZE:
        page = offset // PAGE_SIZE + 1
        page_key = f"compare_page_{page}_{len(matches)}"
        page_col.number_input(
            "Page des variants", 1, (len(matches) + PAGE_SIZE - 1) // PAGE_SIZE, page, step=1, key=page_key,
            on_change=jump_to_page, args=(matches, page_key, "compare_active"),
        )
    ranked = sorted(variants[selected].items(), key=lambda item: (-item[1][0], item[0]))
    with st.container(border=True, key="variant_detail"):
        st.subheader(selected)
        st.write(f"Observé dans **{len(ranked)} catalogues sur {len(sources)}**.")
    table = pd.DataFrame([
        {"Cohorte": source, "Patients avec le variant": patients, "Patients dans la source": sources[source]["patients"],
         "Type tumoral le plus observé": ", ".join(display_cancer(label) for label in labels),
         "Patients dans ce type (chacun si ex æquo)": lead_count}
        for source, (patients, labels, lead_count) in ranked
    ])
    st.dataframe(table, hide_index=True, width="stretch", height=min(450, 39 + 35 * len(table)))
    st.caption("Seuls les catalogues où le variant est rapporté sont affichés. Les panels, les libellés et les patients peuvent se chevaucher ou différer.")
    st.caption("La comparaison porte sur le libellé protéique ; le transcript de référence peut varier selon la source.")
    drill = st.selectbox("Catalogue à examiner en détail", [source for source, _ in ranked])
    path, revision = next((path, revision) for source, path, revision in files if source == drill)
    atlas, _, _ = load_atlas(path, revision)
    render_tumor_breakdown(atlas, selected)


def pick_comparison():
    st.session_state.compare_active = st.session_state.compare_variant


def render_sources():
    st.header("Catalogues et accès")
    try:
        _, sources, _ = load_comparison_index(source_files())
    except ValueError as exc:
        st.error(str(exc))
        return
    rows = [
        {"Cohorte": label, "Patients distincts": row["patients"], "Variants": row["variants"],
         "Types tumoraux": row["types"], "Source": row["url"]}
        for label, row in sorted(sources.items(), key=lambda item: -item[1]["patients"])
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Source": st.column_config.LinkColumn(display_text="Voir la source")})
    else:
        st.info("Aucun catalogue importé.")
    st.caption("Les effectifs sont dédupliqués dans chaque catalogue, pas entre catalogues.")
    st.caption("Les catalogues NCI et ICGC ci-dessus sont issus de données ouvertes, sans compte.")
    if "AACR Project GENIE" not in sources:
        st.info(f"[AACR Project GENIE 20.0]({GENIE}) annonce plus de 242 000 patients. Non inclus : l’accès officiel et le téléchargement des fichiers sont requis.")


def ranked_tumors(atlas, selected):
    return sorted(atlas["alleles"][selected]["cancers"].items(), key=lambda item: (-item[1], item[0]))


def render_allele_summary(atlas, selected):
    record = atlas["alleles"][selected]
    cancers = ranked_tumors(atlas, selected)
    lead_count = cancers[0][1]
    tied = [name for name, carriers in cancers if carriers == lead_count]
    with st.container(key="variant_detail", border=True):
        summary_col, metric_col = st.columns([3, 1], vertical_alignment="center")
        summary_col.caption("VARIANT SÉLECTIONNÉ")
        summary_col.subheader(selected)
        if len(tied) == 1:
            summary_col.markdown(f"Le plus observé : **{display_cancer(tied[0])}** · {count(lead_count)} patients")
        else:
            labels = ", ".join(display_cancer(label) for label in tied)
            summary_col.write(f"**À égalité ({count(lead_count)} chacun) :** {labels}")
        metric_col.metric("Patients avec le variant", count(record["subjects"]))
        metric_col.caption(f"{len(cancers)} types tumoraux")
    if record["subjects"] < 10:
        st.warning("Moins de 10 patients observés : effectif trop faible pour une conclusion fiable.")


def render_tumor_breakdown(atlas, selected):
    cancers = ranked_tumors(atlas, selected)
    st.subheader("Quels cancers ont été observés ?")
    table = pd.DataFrame([
        {"Type tumoral": display_cancer(cancer), "Avec le variant": carriers,
         "Total pour ce cancer": atlas["cancers"][cancer], "Part observée": 100 * carriers / atlas["cancers"][cancer]}
        for cancer, carriers in cancers
    ])
    st.dataframe(
        table, hide_index=True, width="stretch", height=min(355, 39 + 35 * len(table)),
        column_config={
            "Avec le variant": st.column_config.ProgressColumn(
                "Avec le variant", min_value=0, max_value=cancers[0][1], format="%d",
            ),
            "Part observée": st.column_config.NumberColumn("Part observée", format="%.2f %%"),
        },
    )
    with st.expander("Comment lire ces chiffres ?"):
        st.write(
            "La part observée correspond au nombre de patients avec le variant divisé par l’effectif de ce type "
            "dans la source. Les panels de séquençage diffèrent et un patient peut figurer dans plusieurs types. "
            "Le classement par effectif ne démontre ni spécificité du variant ni risque individuel."
        )


def bars(data, category, value, *, color=TEAL, suffix="", x_range=None):
    fig = px.bar(data.iloc[::-1], x=value, y=category, orientation="h", color_discrete_sequence=[color])
    fig.update_traces(hovertemplate=f"%{{y}} · %{{x}}{suffix}<extra></extra>")
    fig.update_layout(
        height=max(250, 36 * len(data) + 54), showlegend=False,
        margin={"l": 4, "r": 18, "t": 10, "b": 10},
        xaxis={"title": value, "range": x_range, "ticksuffix": suffix}, yaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_cohort(run):
    st.subheader("Composition de la cohorte")
    cancer_table = pd.DataFrame(run["cancers"].items(), columns=["Type tumoral", "Dossiers"])
    limit = st.slider("Nombre de types tumoraux affichés", 1, len(cancer_table), min(12, len(cancer_table))) \
        if len(cancer_table) > 1 else 1
    bars(cancer_table.head(limit), "Type tumoral", "Dossiers")

    with st.expander("Voir les effectifs par étude source"):
        st.dataframe(
            pd.DataFrame(run["studies"].items(), columns=["Étude", "Dossiers"]),
            hide_index=True, width="stretch",
        )


def render_models(evaluation):
    st.subheader("Classification hors échantillon")
    st.caption(evaluation["method"])
    if not evaluation["models"]:
        st.info(evaluation.get("unavailable", "Aucun modèle évalué sur ce run."))
        return

    st.write(
        f"**{count(evaluation['patients'])} dossiers évalués**, {len(evaluation['classes'])} classes. "
        "Les signatures alléliques de chaque pli sont apprises uniquement sur sa partie entraînement."
    )
    st.caption(
        f"Métadonnées incohérentes : {count(evaluation['metadata_excluded'])} exclusions. "
        f"Classes trop rares : {count(evaluation['rare_class_excluded'])} dossiers exclus. "
        "Les dossiers sans étiquette ne sont pas évalués."
    )
    metrics = {
        "F1 macro": "f1_macro", "Exactitude équilibrée": "balanced_accuracy",
        "Exactitude globale": "accuracy", "Top 3": "top3_accuracy",
    }
    selected_metric = st.selectbox("Métrique", list(metrics))
    rows = [
        {"Modèle": name, selected_metric: round(100 * values[metrics[selected_metric]], 2)}
        for name, values in evaluation["models"].items() if values[metrics[selected_metric]] is not None
    ]
    if rows:
        bars(pd.DataFrame(rows).sort_values(selected_metric, ascending=False), "Modèle", selected_metric,
             suffix=" %", x_range=[0, 100])
    else:
        st.info("Cette métrique n’est pas disponible pour ce nombre de classes.")
    st.caption("Le F1 macro donne le même poids à chaque classe. Le Top 3 n’est pas une exactitude Top 1.")

    st.subheader("Performance par classe")
    names = list(evaluation["models"])
    model = st.selectbox("Modèle à examiner", names, index=names.index(evaluation["best_model"]))
    result = evaluation["models"][model]
    details = pd.DataFrame(result["per_class"]).rename(columns={
        "cancer": "Type tumoral", "precision": "Précision", "recall": "Rappel", "f1": "F1", "support": "Dossiers",
    })
    st.dataframe(details.round(3), hide_index=True, width="stretch")
    cancer = st.selectbox("Classe réelle à examiner", evaluation["classes"])
    row = result["confusion_matrix"][evaluation["classes"].index(cancer)]
    errors = pd.DataFrame({"Classe prédite": evaluation["classes"], "Dossiers": row})
    bars(errors[errors["Dossiers"] > 0].sort_values("Dossiers", ascending=False).head(10),
         "Classe prédite", "Dossiers", color=INDIGO)
    st.caption("Validation interne sur les cohortes locales ; aucune validation sur une cohorte indépendante.")


def render_external_validation():
    st.subheader("Test sur une autre cohorte")
    if MULTICOHORT_PATH.is_file():
        render_multicohort_validation()
        return
    try:
        with TRANSFER_PATH.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("schema_version") != 1:
            raise ValueError("Format de rapport inconnu.")
    except FileNotFoundError:
        st.info("Aucune évaluation MSK → OrigiMed disponible.")
        st.code("python validate_public_atlas.py", language="powershell")
        return
    except (OSError, ValueError) as exc:
        st.error(f"Rapport illisible : {exc}")
        return
    st.write(
        f"Apprentissage : **{count(report['training_patients'])} patients MSK**. "
        f"Test sans réentraînement : **{count(report['test_patients'])} patients OrigiMed**. "
        f"Classes harmonisées : **{len(report['classes'])}**."
    )
    st.dataframe(pd.DataFrame([
        {"Modèle": name, "F1 macro": percent(result["f1_macro"]),
         "Exactitude équilibrée": percent(result["balanced_accuracy"]), "Exactitude": percent(result["accuracy"])}
        for name, result in report["models"].items()
    ]), hide_index=True, width="stretch")
    st.markdown("**Régression logistique · résultats par type tumoral**")
    st.dataframe(pd.DataFrame([
        {"Type tumoral": row["cancer"], "Patients test": row["support"],
         "Rappel": percent(row["recall"]), "F1": percent(row["f1"])}
        for row in report["models"]["Régression logistique"]["per_class"]
    ]), hide_index=True, width="stretch")
    st.caption(
        "Test de classification après diagnostic, limité aux classes communes et à 50 gènes. "
        "Les panels et pratiques d’annotation des cohortes diffèrent. Ce n’est pas un test de dépistage."
    )


def render_multicohort_validation():
    try:
        with MULTICOHORT_PATH.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("schema_version") != 1:
            raise ValueError("Version de rapport inconnue.")
    except (OSError, ValueError) as exc:
        st.error(f"Rapport multi-cohortes illisible : {exc}")
        return

    st.write(f"**Même test OrigiMed** pour les deux modèles : {count(report['test_patients'])} patients, "
             f"{len(report['classes'])} types tumoraux. OrigiMed n’entre pas dans l’entraînement.")
    reference = report["models"]["Régression logistique · MSK seule"]
    expanded = report["models"]["Régression logistique · Cohortes élargies"]
    gain = 100 * (expanded["f1_macro"] - reference["f1_macro"])
    first, second, third = st.columns(3)
    first.metric("Patients à l’entraînement", count(report["training_patients"]))
    second.metric("F1 macro · MSK seule", percent(reference["f1_macro"]))
    third.metric("F1 macro · Cohortes élargies", percent(expanded["f1_macro"]), f"{gain:+.2f} pts")
    st.dataframe(pd.DataFrame([
        {"Modèle": name, "Patients entraînement": result["training_patients"],
         "F1 macro": percent(result["f1_macro"]), "Exactitude": percent(result["accuracy"]),
         "Exactitude équilibrée": percent(result["balanced_accuracy"])}
        for name, result in report["models"].items()
    ]), hide_index=True, width="stretch")
    st.markdown("**Résultats par type tumoral**")
    st.dataframe(pd.DataFrame([
        {"Type tumoral": new["cancer"], "Patients test": new["support"],
         "F1 · MSK seule": percent(old["f1"]), "F1 · Cohortes élargies": percent(new["f1"]),
         "Rappel · Cohortes élargies": percent(new["recall"])}
        for old, new in zip(reference["per_class"], expanded["per_class"], strict=True)
    ]), hide_index=True, width="stretch")
    with st.expander("Quels jeux ont réellement servi ?"):
        st.dataframe(pd.DataFrame([
            {"Source": row["source"], "Profils lus": row["available"], "Étiquettes compatibles": row["eligible"],
             "Profils attribués": row["used_for_training"]}
            for row in report["sources"]
        ]), hide_index=True, width="stretch")
        st.caption("Un profil rattaché à la première source correspondante peut intégrer des variants d’une autre étude du même réseau.")
        for dataset, reason in report["excluded_catalogs"].items():
            st.caption(f"{dataset} — {reason}")
    st.download_button("Télécharger les résultats agrégés", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="multicohort_validation.json", mime="application/json")
    st.caption("Les profils sont dédupliqués quand les identifiants sont comparables, pas entre tous les réseaux. "
               "La comparaison porte sur la classification de tumeurs déjà diagnostiquées, pas sur le dépistage.")


def render_variants(run):
    st.subheader("Variants rapportés dans le panel")
    st.caption("Un dossier peut contribuer à plusieurs gènes. L’absence de variant rapporté ne prouve pas une couverture suffisante.")
    genes = pd.DataFrame(run["genes"]).rename(columns={"gene": "Gène", "patients": "Dossiers", "variants": "Variants"})
    if genes.empty:
        st.info("Aucun gène dans ce run.")
        return
    metric = st.radio("Trier selon", ["Dossiers", "Variants"], horizontal=True)
    sorted_genes = genes.sort_values(metric, ascending=False, kind="stable")
    bars(sorted_genes.head(15), "Gène", metric)
    st.dataframe(sorted_genes, hide_index=True, width="stretch")


def render_exploration():
    st.header("Cohorte et modèles")
    st.write("Cohortes publiques, variants du panel et comparaison de modèles de classification tumorale.")
    st.caption("Recherche et démonstration uniquement. Pas de diagnostic ni de dépistage.")
    section = st.radio("Section", ["Cohorte", "Modèles", "Variants", "Test externe"], horizontal=True, label_visibility="collapsed")
    if section == "Test externe":
        render_external_validation()
        return
    try:
        revision = SUMMARY_PATH.stat().st_mtime_ns
        run = load_summary(str(SUMMARY_PATH), revision)
    except FileNotFoundError:
        st.info("Aucun run exploratoire disponible. Exécutez le pipeline, puis actualisez cette page.")
        st.code("python main.py --mode real --quick --no-plots", language="powershell")
        st.button("Actualiser")
        return
    except (OSError, ValueError) as exc:
        st.error(f"Lecture du rapport impossible : {exc}")
        return

    evaluation = run["evaluation"]
    created = datetime.fromisoformat(run["created_at"])
    st.sidebar.header("Run exploratoire")
    st.sidebar.caption(f"{created:%d/%m/%Y à %H:%M} UTC · {run['source']}")
    st.sidebar.caption(f"Préparation : {run['preparation_seconds']:.1f} s · total : {run['elapsed_seconds']:.1f} s")
    st.sidebar.button("Actualiser le run", width="stretch")
    st.sidebar.download_button(
        "Exporter les résultats JSON", json.dumps(run, ensure_ascii=False, indent=2),
        file_name="run_summary.json", mime="application/json", width="stretch",
    )
    if run["is_limited"]:
        st.warning("Run limité aux premiers dossiers : la répartition n’est pas représentative de la cohorte complète.")
    best = evaluation["best_model"]
    metrics = st.columns(4)
    metrics[0].metric("Dossiers analysés", count(run["patients"]))
    metrics[1].metric("Variants du panel", count(run["variants"]))
    metrics[2].metric("Études sources", count(len(run["studies"])))
    metrics[3].metric("Meilleur F1 macro · CV", percent(evaluation["models"][best]["f1_macro"]) if best else "—")
    if section == "Cohorte":
        render_cohort(run)
    elif section == "Modèles":
        render_models(evaluation)
    else:
        render_variants(run)


def main():
    st.set_page_config(page_title="Atlas somatique · Allèles et cancers", layout="wide", initial_sidebar_state="collapsed")
    st.html(Path(__file__).with_name(".streamlit") / "atlas.css")
    st.html(
        '<div class="atlas-banner"><p class="atlas-brand">ATLAS SOMATIQUE <span>•</span> RECHERCHE</p>'
        '<h1>Des allèles aux cancers observés</h1>'
        '<p>Choisissez un variant pour voir dans quels types tumoraux il a été rapporté.</p>'
        '<span class="atlas-note">Tumeurs déjà diagnostiquées · Pas une estimation du risque futur</span></div>'
    )
    section = st.radio("Navigation", ["Explorer les allèles", "Cohorte et modèles", "Sources"], horizontal=True,
                       label_visibility="collapsed", key="atlas_nav")
    if section == "Explorer les allèles":
        render_allele_lookup()
    elif section == "Cohorte et modèles":
        render_exploration()
    else:
        render_sources()


if __name__ == "__main__":
    main()
