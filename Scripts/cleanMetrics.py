import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Metrics\metrics-rel-release-2.2.0.csv", low_memory=False)

file_data = data[data["Kind"] == "File"]
entity_data = data[data["Kind"] != "File"]

file_data["BaseFileName"] = file_data["Name"].str.extract(r"(\w+)\.\w+$")

def extract_file_from_entity(entity_name):
    if pd.isna(entity_name) or not isinstance(entity_name, str):
        return None
    
    java_match = re.search(r'\b(\w+)\.\w+$', entity_name)
    if java_match:
        return java_match.group(1)

    hive_match = re.search(r'Hive::(\w+)', entity_name)
    if hive_match:
        return hive_match.group(1)

    thrift_match = re.search(r'(\w+)::\w+$', entity_name)
    if thrift_match:
        return thrift_match.group(1)

    return None

entity_data["BaseFileName"] = entity_data["Name"].apply(extract_file_from_entity)

entity_data = entity_data.merge(file_data[["BaseFileName", "Name"]], on="BaseFileName", how="left", suffixes=("", "_File"))

valid_entity_data = entity_data[entity_data["Name_File"].notna()]

aggregated_metrics = valid_entity_data.groupby("BaseFileName").agg({
    "Cyclomatic": ["mean", "sum", "max"],
    "CountLineCode": ["mean", "sum", "max"],
    "MaxNesting": "max",
    "CountDeclMethod": "sum"
}).reset_index()

aggregated_metrics.columns = [
    "BaseFileName", "AvgCyclomatic", "SumCyclomatic", "MaxCyclomatic",
    "AvgLineCode", "SumLineCode", "MaxLineCode", "MaxNesting", "SumDeclMethod"
]

correlation_matrix = aggregated_metrics.corr(method='spearman').abs()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation (Spearman)")
plt.show()

threshold = 0.7
columns_to_remove = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > threshold:
            colname = correlation_matrix.columns[i]
            if colname not in columns_to_remove:
                columns_to_remove.append(colname)

print(f"Colonnes supprimées en raison d'une forte corrélation : {columns_to_remove}")
aggregated_metrics.drop(columns=columns_to_remove, inplace=True)

aggregated_metrics.fillna(aggregated_metrics.mean(), inplace=True)

print("Statistiques des métriques agrégées :")
print(aggregated_metrics.describe())

file_data = file_data.merge(aggregated_metrics, on="BaseFileName", how="left")

file_data.fillna(0, inplace=True)

print("Données après fusion et agrégation :")
print(file_data.head())

output_path = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Clean_Metrics\result-metrics-rel-release-2.2.0.csv"
file_data.to_csv(output_path, index=False)