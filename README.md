# MGL869_LAB
Utilisation de l’apprentissage machine pour prédiction des bogues

## Architecture du dépot
- [CSV_files](https://github.com/GregPtto/MGL869_LAB/tree/main/CSV_files) => Contient tous les résultats obtenus
    - [Clean_Metrics](https://github.com/GregPtto/MGL869_LAB/tree/main/CSV_files/Clean_Metrics) => Fichiers csv des métriques labelisés
    - [Metrics](https://github.com/GregPtto/MGL869_LAB/tree/main/CSV_files/Metrics) => Fichiers csv d'analyse Understand
    - [Results](https://github.com/GregPtto/MGL869_LAB/tree/main/CSV_files/Results) => Dossiers pour toutes les versions mineures contenant les analyses de performance
        - ... => Fichiers d'analyse avec Readme.md les résultats 
        - [PERFORMANCE.md](https://github.com/GregPtto/MGL869_LAB/tree/main/CSV_files/Results/PERFORMANCE.md) => Rapport des métriques par versions mineures
- [Scripts](https://github.com/GregPtto/MGL869_LAB/tree/main/Scripts) => Contient tous les scripts python du Laboratoire

## Collection des données
Script de collections des données :
- [fetchJiraData.py](https://github.com/GregPtto/MGL869_LAB/blob/main/Scripts/fetchJiraData.py) => Récupère les bogues depuis Jira
    - jira_issues.csv
- [bugFileAssociation.py](https://github.com/GregPtto/MGL869_LAB/blob/main/Scripts/bugFileAssociation.py) => Associe les bogues au dernier commit de leur version
    - bug_file_association.csv
- [metricsCollection.py](https://github.com/GregPtto/MGL869_LAB/blob/main/Scripts/metricsCollection.py) => Analyse Understand pour les fichiers du dernier commit de chaque version
    - CSV_files/Metrics/...
- [cleanMetrics.py](https://github.com/GregPtto/MGL869_LAB/blob/main/Scripts/cleanMetrics.py) => Nettoyage des données et aggrégation de certaines variables indépendantes
    - CSV_files/Clean_Metrics/...
- [dataLabelisation.py](https://github.com/GregPtto/MGL869_LAB/blob/main/Scripts/dataLabelisation.py) => Labelisation des lignes en fonction des données Jira Bogue/Non-Bogue
    - CSV_files/Clean_Metrics/...

## Modèles de prédiction

- [modele2.py](https://github.com/GregPtto/MGL869_LAB/tree/main/Script/modele2.py) => Construit nos deux modèles (Logistic Regression et Random Forest) et effectue les entraînements ainsi que les analyse pour chaque version mineurs. Génère également les tableaux et matrics d'analyse de performance résumé dans un fichier Readme qui met en comparaison les deux modèles.
    - CSV_files/Results/...

- [showPerformance.py](https://github.com/GregPtto/MGL869_LAB/tree/main/Scripts/showPerformance.py) => Construit les graphiques de performance pour les trois métriques (AUC, Précision, Recall) et génère le fichier PERFORMANCE.md
    - [PERFORMANCE.md](https://github.com/GregPtto/MGL869_LAB/tree/main/CSV_files/Results/PERFORMANCE.md) => Comparaison des 3 métriques de performance (AUC, Précision, Recall) pour les deux modèles.

## Rapport du Laboratoire
[Lien du rapport](https://etsmtl365-my.sharepoint.com/:w:/r/personal/gregory_pititto_1_ens_etsmtl_ca/Documents/MGL899-LAB-Rapport.docx?d=w8efb0f1954a64386858b572e90b4d18a&csf=1&web=1&e=0KeAcU)