import pandas as pd
import re


# Charger les données
data = pd.read_csv(r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Metrics\metrics-release-2.0.0.csv", low_memory=False)

# Étape 1 : Séparer les fichiers et les entités
file_data = data[data["Kind"] == "File"]  # Fichiers
entity_data = data[data["Kind"] != "File"]  # Entités (méthodes, classes, etc.)

# Vérifier les données séparées
if file_data.empty or entity_data.empty:
    print("Attention : 'file_data' ou 'entity_data' est vide. Vérifiez vos données.")
    exit()

# Étape 2 : Extraire le nom des fichiers sans extension (pour la colonne 'Name')
file_data["BaseFileName"] = file_data["Name"].str.extract(r"(\w+)\.\w+$")  # Extrait 'AMReporter' de 'AMReporter.java'

# Étape 3 : Extraire le fichier parent des noms qualifiés dans 'entity_data'
def extract_file_from_entity(entity_name):
    """
    Extrait le nom de fichier (sans extension) à partir d'un nom qualifié d'entité.
    Exemple : 'org.apache.hadoop.hive.llap.daemon.impl.AMReporter.AMNodeInfo.getDelay'
    Retourne : 'AMReporter'
    """
    if pd.isna(entity_name) or not isinstance(entity_name, str):
        return None
    # Cas Java : capturer la partie avant le dernier '.'
    java_match = re.search(r'\b(\w+)\.\w+$', entity_name)
    if java_match:
        return java_match.group(1)

    # Cas C++ ou Thrift avec 'Hive::' spécifique
    hive_match = re.search(r'Hive::(\w+)', entity_name)
    if hive_match:
        return hive_match.group(1)

    # Cas Thrift ou C++ général : capturer le segment avant le dernier '::'
    thrift_match = re.search(r'(\w+)::\w+$', entity_name)
    if thrift_match:
        return thrift_match.group(1)

    return None

# Appliquer la fonction pour extraire les noms de fichiers dans 'entity_data'
entity_data["BaseFileName"] = entity_data["Name"].apply(extract_file_from_entity)

# Étape 4 : Associer les entités aux fichiers
entity_data = entity_data.merge(file_data[["BaseFileName", "Name"]], on="BaseFileName", how="left", suffixes=("", "_File"))

# Filtrer les entités qui ont une correspondance avec un fichier (ignorer celles sans correspondance)
valid_entity_data = entity_data[entity_data["Name_File"].notna()]

# Vérification des entités associées
print("Exemples d'entités associées à des fichiers :")
print(valid_entity_data[["Name", "BaseFileName", "Name_File"]].dropna().head(10))

# Étape 5 : Agrégation des métriques par fichier
aggregated_metrics = valid_entity_data.groupby("BaseFileName").agg({
    "Cyclomatic": ["mean", "sum", "max"],
    "CountLineCode": ["mean", "sum", "max"],
    "MaxNesting": "max",
    "CountDeclMethod": "sum"
}).reset_index()

# Aplatir les colonnes multi-niveaux après l'agrégation
aggregated_metrics.columns = [
    "BaseFileName", "AvgCyclomatic", "SumCyclomatic", "MaxCyclomatic",
    "AvgLineCode", "SumLineCode", "MaxLineCode", "MaxNesting", "SumDeclMethod"
]

# Étape 6 : Imputation des valeurs manquantes après l'agrégation
# Remplir les valeurs manquantes par la moyenne de chaque colonne
aggregated_metrics.fillna(aggregated_metrics.mean(), inplace=True)

# Vérification des métriques agrégées
print("Statistiques des métriques agrégées :")
print(aggregated_metrics.describe())

# Étape 7 : Fusionner les métriques agrégées avec les fichiers
file_data = file_data.merge(aggregated_metrics, on="BaseFileName", how="left")

# Remplacer les NaN par 0 pour les fichiers sans entités associées
file_data.fillna(0, inplace=True)

# Vérifier les résultats finaux
print("Données après fusion et agrégation :")
print(file_data.head())

# Sauvegarder les résultats
output_path = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Clean_Metrics\result-metrics-release-2.0.0.csv"
file_data.to_csv(output_path, index=False)
print(f"Données nettoyées et agrégées sauvegardées dans '{output_path}'")