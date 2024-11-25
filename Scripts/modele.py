import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from openpyxl import Workbook

# Charger les données
input_path = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Clean_Metrics\result-metrics-release-2.0.0.csv"
data = pd.read_csv(input_path, low_memory=False)

# Étape 1 : Nettoyage des données
print("Nettoyage des données...")
# Convertir les virgules en points dans les colonnes numériques (si applicable)
data.replace({",": "."}, regex=True, inplace=True)
data = data.apply(pd.to_numeric, errors="ignore")

# Supprimer les colonnes non pertinentes
columns_to_remove = ["Kind", "Name", "Version", "CommitId"]  # Ajustez si nécessaire
data = data.drop(columns=columns_to_remove, errors="ignore")

# Séparer les features (X) et la cible (y)
X = data.drop(columns=["Bogue"], errors="ignore")
y = data["Bogue"]

# Supprimer les colonnes non numériques
X = X.select_dtypes(include=[np.number])

# Vérification des dimensions
print(f"Dimensions après nettoyage : X={X.shape}, y={y.shape}")

# Étape 2 : Standardisation
print("Standardisation des données...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 3 : Définir les modèles
print("Initialisation des modèles...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Étape 4 : Définir la validation croisée
print("Configuration de la validation croisée...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Étape 5 : Validation croisée et stockage des résultats
results = []
output_path = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\model_results.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for model_name, model in models.items():
        print(f"En cours d'évaluation : {model_name}")
        
        # Effectuer la validation croisée
        cv_results = cross_validate(
            model, X_scaled, y, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False
        )
        
        # Création d'un DataFrame des résultats pour un modèle
        model_results = pd.DataFrame({
            "Fold": range(1, len(cv_results["test_accuracy"]) + 1),
            "Accuracy": cv_results["test_accuracy"],
            "Precision": cv_results["test_precision"],
            "Recall": cv_results["test_recall"],
            "F1 Score": cv_results["test_f1"],
            "ROC AUC": cv_results["test_roc_auc"]
        })
        
        # Calculer les moyennes des métriques
        summary = model_results.mean().to_dict()
        summary["Model"] = model_name
        results.append(summary)
        
        # Sauvegarder les résultats détaillés de chaque modèle dans un onglet Excel
        model_results.to_excel(writer, index=False, sheet_name=model_name[:30])  # Limite à 30 caractères pour les noms d'onglet Excel

# Sauvegarder le résumé des modèles dans un onglet séparé
results_df = pd.DataFrame(results)
results_df.to_excel(writer, index=False, sheet_name="Summary")

print(f"Les résultats ont été sauvegardés dans : {output_path}")
