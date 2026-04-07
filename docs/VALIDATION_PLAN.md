# Validation Plan

## Objectif

Faire passer le projet d'un prototype de recherche a un systeme evaluable de
maniere rigoureuse.

## Phase 1 - Validation technique

- figer version code, config, features et datasets
- enregistrer un manifeste de run et un manifeste de datasets
- tester la reproductibilite inter-run
- ajouter des tests d'integration sur mini cohortes fixes
- documenter les cas invalides et les rejets

## Phase 2 - Validation scientifique

- separer strictement train / validation interne / test externe
- definir a l'avance les metriques cibles
- analyser calibration, erreurs, classes rares et biais de population
- comparer aux baseline simples et a un standard de reference
- documenter les intervalles de confiance

## Phase 3 - Validation clinique

- definir une claim clinique precise
- collecter cohortes externes independantes
- rediger protocole avant analyse
- evaluer impact des faux positifs / faux negatifs
- produire rapport clinique par sous-groupes

## Artefacts attendus

- manifestes de datasets
- manifestes de run
- rapports de performance verrouilles
- registre de risques
- journal des changements du modele
