"""
Dashboard interactif — DNA Cancer Analysis Pipeline
Lancement : python -m streamlit run dashboard.py
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DNA Cancer Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORTS_DIR = "output/reports"
CSV_PATH    = os.path.join(REPORTS_DIR, "rapport_cohorte.csv")
JSON_PATH   = os.path.join(REPORTS_DIR, "ml_results.json")
MATRIX_PATH = os.path.join(REPORTS_DIR, "mutation_matrix.json")

RISK_COLORS = {"FAIBLE": "#2ecc71", "MODERE": "#f39c12", "ELEVE": "#e74c3c", "CRITIQUE": "#8e44ad"}
MODEL_COLORS = {
    "Logistic Regression": "#3498db",
    "Random Forest":       "#2ecc71",
    "Gradient Boosting":   "#e67e22",
    "SVM (RBF)":           "#9b59b6",
    "LightGBM":            "#1abc9c",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Chargement des données (cache)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Confiance_ML"]      = pd.to_numeric(df["Confiance_ML"], errors="coerce")
    df["Total_Mutations"]   = pd.to_numeric(df["Total_Mutations"], errors="coerce")
    df["Age"]               = pd.to_numeric(df["Age"], errors="coerce")
    df["Cancer_ML_Correct"] = df["Cancer_ML_Correct"].map({"True": True, "False": False})
    return df

@st.cache_data
def load_ml() -> dict:
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_matrix() -> dict:
    with open(MATRIX_PATH, encoding="utf-8") as f:
        return json.load(f)["matrix"]

def files_ready() -> bool:
    return all(os.path.exists(p) for p in [CSV_PATH, JSON_PATH])

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers graphiques
# ─────────────────────────────────────────────────────────────────────────────
def kpi(col, label: str, value: str, delta: str = "", color: str = "#3498db"):
    col.markdown(
        f"""
        <div style="background:#1e1e2e;border-left:4px solid {color};
                    padding:14px 18px;border-radius:6px;margin-bottom:4px">
            <div style="font-size:12px;color:#aaa;margin-bottom:4px">{label}</div>
            <div style="font-size:26px;font-weight:700;color:white">{value}</div>
            <div style="font-size:12px;color:#aaa">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def badge(correct: bool) -> str:
    if correct is True:
        return "✅"
    if correct is False:
        return "❌"
    return "—"

# ─────────────────────────────────────────────────────────────────────────────
#  Garde-fou — pipeline pas encore lancé
# ─────────────────────────────────────────────────────────────────────────────
if not files_ready():
    st.title("🧬 DNA Cancer Analysis — Dashboard")
    st.warning(
        "Les fichiers de résultats sont introuvables. "
        "Lance d'abord le pipeline : `python main.py`",
        icon="⚠️",
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  Chargement
# ─────────────────────────────────────────────────────────────────────────────
df  = load_csv()
ml  = load_ml()

# ─────────────────────────────────────────────────────────────────────────────
#  Entête
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧬 DNA Cancer Analysis — Dashboard")
st.caption(
    f"Données TCGA PanCancer Atlas · {ml.get('n_samples_labeled', len(df))} patients "
    f"· {len(ml.get('class_distribution', {}))} types de cancer"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'ensemble",
    "🎯 Performance par cancer",
    "🔬 Variants discriminants",
    "👤 Fiche patient",
    "🧪 Signatures allèles",
])

# ═════════════════════════════════════════════════════════════════════════════
#  ONGLET 1 — Vue d'ensemble
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    best_model  = ml.get("best_model", "—")
    f1_macro    = ml.get("best_f1_macro", 0)
    n_labeled   = ml.get("n_samples_labeled", len(df))
    top3_acc    = ml.get("models", {}).get(best_model, {}).get("top3_accuracy") or 0
    acc_overall = df["Cancer_ML_Correct"].mean() if "Cancer_ML_Correct" in df else 0

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Patients analysés",   f"{n_labeled:,}",        "TCGA PanCancer Atlas",            "#3498db")
    kpi(c2, "Meilleur modèle",     best_model,              f"f1_macro = {f1_macro:.3f}",       "#e67e22")
    kpi(c3, "Accuracy globale",    f"{acc_overall:.1%}",    "sur patients labellisés",          "#2ecc71")
    kpi(c4, "Top-3 accuracy",      f"{top3_acc:.1%}",       "vrai type dans le top-3",          "#9b59b6")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    # Distribution des cancers
    with col_left:
        st.subheader("Distribution des cancers")
        dist = pd.DataFrame(
            list(ml["class_distribution"].items()),
            columns=["Cancer", "N_patients"]
        ).sort_values("N_patients", ascending=True)
        fig = px.bar(
            dist, x="N_patients", y="Cancer", orientation="h",
            color="N_patients", color_continuous_scale="Blues",
            labels={"N_patients": "Patients", "Cancer": ""},
            height=600,
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Comparaison des modèles
    with col_right:
        st.subheader("Comparaison des modèles")
        model_rows = []
        for name, stats in ml.get("models", {}).items():
            if name == "Baseline (majority class)":
                continue
            model_rows.append({
                "Modèle":     name,
                "Accuracy":   stats.get("accuracy", 0),
                "F1-macro":   stats.get("f1", 0),
                "AUC":        stats.get("roc_auc") or 0,
                "Top-3":      stats.get("top3_accuracy") or 0,
            })
        model_df = pd.DataFrame(model_rows)
        fig2 = go.Figure()
        metrics = ["Accuracy", "F1-macro", "AUC", "Top-3"]
        colors  = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
        for metric, color in zip(metrics, colors):
            fig2.add_trace(go.Bar(
                name=metric, x=model_df["Modèle"], y=model_df[metric],
                marker_color=color, opacity=0.85,
            ))
        fig2.update_layout(
            barmode="group", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Tableau récapitulatif")
        display_df = model_df.copy()
        for col in ["Accuracy", "F1-macro", "AUC", "Top-3"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")
        best_idx = model_df["F1-macro"].idxmax()
        st.dataframe(
            display_df.style.apply(
                lambda row: ["background-color:#2c3e50" if row.name == best_idx else "" for _ in row],
                axis=1,
            ),
            use_container_width=True, hide_index=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
#  ONGLET 2 — Performance par cancer
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    # Calculer métriques par cancer depuis le CSV
    labeled = df[df["Cancer_Connu"].notna() & (df["Cancer_Connu"] != "")]
    per_cancer = (
        labeled.groupby("Cancer_Connu")
        .agg(
            N=("Patient_ID", "count"),
            Correct=("Cancer_ML_Correct", "sum"),
            Conf_moy=("Confiance_ML", "mean"),
        )
        .reset_index()
    )
    per_cancer["Accuracy"] = per_cancer["Correct"] / per_cancer["N"]

    # F1 par cancer depuis rapport_ml.txt
    f1_map = {}
    try:
        with open(os.path.join(REPORTS_DIR, "rapport_ml.txt"), encoding="utf-8") as f:
            lines = f.readlines()
        in_class = False
        for line in lines:
            if "Classe" in line and "Prec" in line:
                in_class = True
                continue
            if in_class:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        cancer_name = parts[0]
                        f1_val = float(parts[3])
                        f1_map[cancer_name] = f1_val
                    except ValueError:
                        if not parts[0][0].isalpha():
                            in_class = False
    except Exception:
        pass

    per_cancer["F1"] = per_cancer["Cancer_Connu"].map(f1_map).fillna(0)
    per_cancer = per_cancer.sort_values("Accuracy", ascending=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Accuracy par type de cancer")
        colors_acc = ["#e74c3c" if v < 0.6 else "#2ecc71" for v in per_cancer["Accuracy"]]
        fig_acc = px.bar(
            per_cancer, x="Accuracy", y="Cancer_Connu", orientation="h",
            text=per_cancer["Accuracy"].map(lambda x: f"{x:.0%}"),
            custom_data=["N"],
            labels={"Cancer_Connu": "", "Accuracy": "Accuracy"},
            height=600,
        )
        fig_acc.update_traces(marker_color=colors_acc, textposition="outside")
        fig_acc.update_layout(xaxis_range=[0, 1.1], margin=dict(l=0, r=0, t=10, b=0))
        fig_acc.update_traces(hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.1%}<br>N: %{customdata[0]}<extra></extra>")
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_b:
        st.subheader("F1-score par type de cancer")
        pc_f1 = per_cancer.sort_values("F1", ascending=True)
        colors_f1 = ["#e74c3c" if v < 0.4 else "#2ecc71" for v in pc_f1["F1"]]
        fig_f1 = px.bar(
            pc_f1, x="F1", y="Cancer_Connu", orientation="h",
            text=pc_f1["F1"].map(lambda x: f"{x:.3f}"),
            labels={"Cancer_Connu": "", "F1": "F1-score"},
            height=600,
        )
        fig_f1.update_traces(marker_color=colors_f1, textposition="outside")
        fig_f1.update_layout(xaxis_range=[0, 1.1], margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("---")
    st.subheader("Matrice de confusion interactive")

    # Construire matrice de confusion depuis CSV
    known = labeled[labeled["Cancer_ML_Predit"].notna()]
    cancers = sorted(known["Cancer_Connu"].unique())
    cm_df   = pd.crosstab(known["Cancer_Connu"], known["Cancer_ML_Predit"])
    cm_df   = cm_df.reindex(index=cancers, columns=cancers, fill_value=0)
    cm_arr  = cm_df.values

    norm_type = st.radio("Normalisation", ["Absolue", "Par ligne (recall)"], horizontal=True)
    if norm_type == "Par ligne (recall)":
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_plot  = np.divide(cm_arr, row_sums, where=row_sums != 0)
        fmt      = ".2f"
    else:
        cm_plot = cm_arr
        fmt     = "d"

    fig_cm = px.imshow(
        cm_plot,
        x=cancers, y=cancers,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Réel", color=""),
        aspect="auto",
        text_auto=fmt,
    )
    fig_cm.update_layout(
        height=700,
        xaxis_tickangle=-45,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
#  ONGLET 3 — Variants discriminants
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    # Aplatir top_discriminant_alleles
    rows_alleles = []
    for cancer, alleles in ml.get("top_discriminant_alleles", {}).items():
        for a in alleles:
            rows_alleles.append({
                "Cancer":         a.get("cancer", cancer),
                "Gène":           a.get("gene", ""),
                "Variant":        a.get("allele", ""),
                "Enrichissement": a.get("enrichment", 0),
                "Odds Ratio":     a.get("odds_ratio", 0),
                "Fréq. in":       a.get("freq_in_cancer", 0),
                "Fréq. out":      a.get("freq_outside_cancer", 0),
                "N patients":     a.get("n_patients_with", 0),
                "Hotspot":        "✅" if a.get("is_hotspot") else "—",
            })

    allele_df = pd.DataFrame(rows_alleles).sort_values("Enrichissement", ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"{len(allele_df)} variants discriminants identifiés")

        cancer_filter = st.multiselect(
            "Filtrer par cancer", sorted(allele_df["Cancer"].unique()),
            default=[], key="allele_cancer_filter"
        )
        df_filt = allele_df[allele_df["Cancer"].isin(cancer_filter)] if cancer_filter else allele_df

        st.dataframe(
            df_filt.style.format({
                "Enrichissement": lambda x: f"{x:,.0f}×" if x > 100 else f"{x:.2f}×",
                "Fréq. in":       "{:.1%}",
                "Fréq. out":      "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

    with col2:
        st.subheader("Top 15 variants — enrichissement (log)")
        top15 = allele_df.head(15).copy()
        top15["Label"] = top15["Gène"] + " " + top15["Variant"]
        top15 = top15.sort_values("Enrichissement", ascending=True)
        fig_en = px.bar(
            top15, x="Enrichissement", y="Label", orientation="h",
            color="Cancer",
            text=top15["Enrichissement"].map(lambda x: f"{x:,.0f}×"),
            log_x=True,
            labels={"Label": "", "Enrichissement": "Enrichissement (log)"},
            height=500,
        )
        fig_en.update_traces(textposition="outside")
        fig_en.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=True)
        st.plotly_chart(fig_en, use_container_width=True)

    st.markdown("---")
    st.subheader("Détail par cancer")
    selected_cancer = st.selectbox(
        "Choisir un cancer", sorted(allele_df["Cancer"].unique()), key="detail_cancer"
    )
    cancer_alleles = allele_df[allele_df["Cancer"] == selected_cancer]
    if cancer_alleles.empty:
        st.info("Aucun variant discriminant pour ce cancer (panel 26 gènes insuffisant).")
    else:
        n_pat = ml["class_distribution"].get(selected_cancer, "?")
        st.caption(f"{n_pat} patients · {len(cancer_alleles)} variant(s) discriminant(s)")
        st.dataframe(
            cancer_alleles.style.format({
                "Enrichissement": lambda x: f"{x:,.0f}×" if x > 100 else f"{x:.2f}×",
                "Fréq. in":       "{:.1%}",
                "Fréq. out":      "{:.2%}",
            }),
            use_container_width=True, hide_index=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
#  ONGLET 4 — Fiche patient
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.sidebar.markdown("## Filtres patients")

    # Filtres sidebar
    all_cancers = sorted(df["Cancer_Connu"].dropna().unique())
    sel_cancers = st.sidebar.multiselect("Cancer connu", all_cancers, key="filter_cancer")

    all_sexes = sorted(df["Sexe"].dropna().unique())
    sel_sexes = st.sidebar.multiselect("Sexe", all_sexes, key="filter_sexe")

    correct_opt = st.sidebar.radio(
        "Prédiction ML", ["Tous", "Correctes ✅", "Incorrectes ❌"], key="filter_correct"
    )

    conf_min, conf_max = st.sidebar.slider(
        "Confiance ML", 0.0, 1.0, (0.0, 1.0), 0.05, key="filter_conf"
    )

    risk_opts = st.sidebar.multiselect(
        "Risque global", ["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"], key="filter_risk"
    )

    search_id = st.sidebar.text_input("Recherche par ID patient (ex: PAT_0001)", key="search_id")

    # Appliquer filtres
    fdf = df.copy()
    if sel_cancers:
        fdf = fdf[fdf["Cancer_Connu"].isin(sel_cancers)]
    if sel_sexes:
        fdf = fdf[fdf["Sexe"].isin(sel_sexes)]
    if correct_opt == "Correctes ✅":
        fdf = fdf[fdf["Cancer_ML_Correct"]]
    elif correct_opt == "Incorrectes ❌":
        fdf = fdf[~fdf["Cancer_ML_Correct"]]
    fdf = fdf[fdf["Confiance_ML"].between(conf_min, conf_max, inclusive="both")]
    if risk_opts:
        fdf = fdf[fdf["Risque_Global"].isin(risk_opts)]
    if search_id.strip():
        fdf = fdf[fdf["Patient_ID"].str.contains(search_id.strip(), case=False, na=False)]

    st.subheader(f"{len(fdf):,} patient(s) — sélectionne une ligne pour voir le détail")

    # Tableau
    display_cols = ["Patient_ID", "Age", "Sexe", "Cancer_Connu", "Cancer_ML_Predit",
                    "Confiance_ML", "Cancer_ML_Correct", "Risque_Global",
                    "Total_Mutations", "N_Hotspots"]
    table_df = fdf[display_cols].copy()
    table_df["Cancer_ML_Correct"] = table_df["Cancer_ML_Correct"].map(
        {True: "✅", False: "❌"}
    ).fillna("—")

    event = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=320,
        column_config={
            "Confiance_ML": st.column_config.ProgressColumn(
                "Confiance ML", min_value=0, max_value=1, format="%.2f"
            ),
        },
    )

    # Panneau détail
    selected_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
    if selected_rows:
        idx   = fdf.index[selected_rows[0]]
        pat   = df.loc[idx]
        pid   = pat["Patient_ID"]

        st.markdown("---")
        st.subheader(f"Fiche patient — {pid}")

        info_col, pred_col, mut_col = st.columns([1, 1, 2])

        with info_col:
            st.markdown("**Informations cliniques**")
            st.metric("Âge", f"{int(pat['Age'])} ans" if pd.notna(pat["Age"]) else "—")
            st.metric("Sexe", pat["Sexe"])
            st.metric("Cancer connu", pat["Cancer_Connu"])
            st.metric("Sévérité", pat.get("Severite", "—"))
            risk = pat.get("Risque_Global", "—")
            color = RISK_COLORS.get(risk, "#aaa")
            st.markdown(
                f'Risque global : <span style="color:{color};font-weight:700">{risk}</span>',
                unsafe_allow_html=True,
            )

        with pred_col:
            st.markdown("**Prédiction ML**")
            correct = pat["Cancer_ML_Correct"]
            icon = "✅" if correct is True else "❌" if correct is False else "—"
            conf = pat["Confiance_ML"]
            st.metric("Cancer prédit", f"{pat['Cancer_ML_Predit']} {icon}")
            st.metric("Confiance", f"{conf:.1%}" if pd.notna(conf) else "—")
            if pd.notna(pat.get("Top2_Cancer")):
                st.metric("2e choix", pat["Top2_Cancer"])
            if pd.notna(pat.get("Top3_Cancer")):
                st.metric("3e choix", pat["Top3_Cancer"])
            st.markdown("**Stats mutationnelles**")
            st.metric("Mutations totales", int(pat["Total_Mutations"]) if pd.notna(pat["Total_Mutations"]) else "—")
            st.metric("Hotspots", int(pat["N_Hotspots"]) if pd.notna(pat["N_Hotspots"]) else "—")
            st.metric("Variants pathogènes", int(pat["N_Pathogeniques"]) if pd.notna(pat["N_Pathogeniques"]) else "—")

        with mut_col:
            st.markdown("**Profil mutationnel (26 gènes)**")
            try:
                matrix = load_matrix()
                if pid in matrix:
                    gene_counts = matrix[pid]
                    gene_df = pd.DataFrame(
                        [{"Gène": g, "Mutations": v} for g, v in sorted(gene_counts.items())
                         if isinstance(v, (int, float))]
                    ).sort_values("Mutations", ascending=True)
                    gene_df = gene_df[gene_df["Mutations"] >= 0]

                    fig_mut = px.bar(
                        gene_df, x="Mutations", y="Gène", orientation="h",
                        color="Mutations",
                        color_continuous_scale=[[0, "#1a1a2e"], [0.01, "#3498db"], [1, "#e74c3c"]],
                        labels={"Gène": "", "Mutations": "Nb mutations"},
                        height=450,
                    )
                    fig_mut.update_layout(
                        coloraxis_showscale=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig_mut, use_container_width=True)
                else:
                    st.info("Profil mutationnel non disponible pour ce patient.")
            except FileNotFoundError:
                st.info("mutation_matrix.json introuvable.")
    else:
        st.info("Sélectionne une ligne dans le tableau pour voir la fiche complète du patient.")

# ═════════════════════════════════════════════════════════════════════════════
#  ONGLET 5 — Signatures allèles
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Signatures allèles par type de cancer")
    st.caption("Variants somatiques enrichis dans chaque type tumoral (panel 26 gènes, cohorte TCGA)")

    sigs = ml.get("allele_signatures", {})
    disc = ml.get("top_discriminant_alleles", {})

    sig_rows = []
    for cancer, info in sigs.items():
        n_alleles  = info.get("n_alleles", 0)
        n_patients = info.get("n_patients", 0)
        variants   = disc.get(cancer, [])
        top_var    = ", ".join(
            f"{v['gene']} {v['allele']} ({v['enrichment']:.0f}×)"
            for v in sorted(variants, key=lambda x: x["enrichment"], reverse=True)[:3]
        ) or "Aucun variant discriminant"
        sig_rows.append({
            "Cancer":             cancer,
            "Patients":           n_patients,
            "Variants discrim.":  n_alleles,
            "Top variants":       top_var,
        })

    sig_df = pd.DataFrame(sig_rows).sort_values("Variants discrim.", ascending=False)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.dataframe(sig_df, use_container_width=True, hide_index=True, height=500)

    with col_right:
        fig_sig = px.bar(
            sig_df.sort_values("Variants discrim.", ascending=True),
            x="Variants discrim.", y="Cancer", orientation="h",
            color="Variants discrim.",
            color_continuous_scale="Teal",
            text="Variants discrim.",
            labels={"Cancer": "", "Variants discrim.": "Variants discriminants"},
            height=500,
        )
        fig_sig.update_traces(textposition="outside")
        fig_sig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("---")
    st.subheader("Détail — Sélectionner un cancer")

    cancers_with_sigs = [c for c, v in disc.items() if v]
    if cancers_with_sigs:
        sel_sig = st.selectbox("Cancer", sorted(cancers_with_sigs), key="sig_select")
        sig_alleles = disc.get(sel_sig, [])
        n_pat_sig   = ml["class_distribution"].get(sel_sig, "?")

        st.caption(f"{n_pat_sig} patients dans la cohorte · {len(sig_alleles)} variant(s)")

        sig_detail = pd.DataFrame([{
            "Gène":           a["gene"],
            "Variant":        a["allele"],
            "Enrichissement": a["enrichment"],
            "Fréq. cancer":   a["freq_in_cancer"],
            "Fréq. autres":   a["freq_outside_cancer"],
            "N porteurs":     a["n_patients_with"],
            "N total cancer": a["n_cancer_total"],
            "Hotspot":        "✅" if a.get("is_hotspot") else "—",
        } for a in sorted(sig_alleles, key=lambda x: x["enrichment"], reverse=True)])

        st.dataframe(
            sig_detail.style.format({
                "Enrichissement": lambda x: f"{x:,.0f}×" if x > 100 else f"{x:.2f}×",
                "Fréq. cancer":   "{:.1%}",
                "Fréq. autres":   "{:.2%}",
            }),
            use_container_width=True, hide_index=True,
        )

        # Graphique enrichissement pour ce cancer
        if len(sig_detail) > 0:
            fig_detail = px.bar(
                sig_detail.sort_values("Enrichissement"),
                x="Enrichissement", y=sig_detail.apply(lambda r: f"{r['Gène']} {r['Variant']}", axis=1),
                orientation="h", log_x=True,
                text=sig_detail.sort_values("Enrichissement")["Enrichissement"].map(
                    lambda x: f"{x:,.0f}×" if x > 100 else f"{x:.2f}×"
                ),
                color_discrete_sequence=["#e67e22"],
                labels={"y": "", "Enrichissement": "Enrichissement (log)"},
                height=max(200, len(sig_detail) * 60),
            )
            fig_detail.update_traces(textposition="outside")
            fig_detail.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_detail, use_container_width=True)
    else:
        st.info("Aucun variant discriminant identifié dans la cohorte actuelle.")

# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "DNA Cancer Analysis Pipeline · Données TCGA PanCancer Atlas · "
    "Projet ING2 · Lancer le pipeline : `python main.py`"
)
