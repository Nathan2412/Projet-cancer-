# Risk Register

## Risques majeurs

1. Mauvaise generalisation hors cohortes publiques.
   Mitigation: validation externe independante, suivi post-deploiement.

2. Sur-interpretation clinique d'une prediction faible confiance.
   Mitigation: seuil d'incertitude, affichage explicite "prediction incertaine".

3. Desequilibre severe entre classes tumorales.
   Mitigation: metriques robustes, renforcement des classes rares, analyse par classe.

4. Heterogeneite des cohortes sources.
   Mitigation: manifeste datasets, harmonisation labels, analyses par etude.

5. Derive de distribution sur de nouvelles donnees.
   Mitigation: surveillance, journalisation, revalidation avant re-entrainement.

6. Regles cliniques incoherentes.
   Mitigation: centralisation dans `clinical_rules.py` et tests de non-regression.
