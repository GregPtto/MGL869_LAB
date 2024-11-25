# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, classification_report, roc_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Charger vos données depuis le chemin fourni
file_path = r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Clean_Metrics\result-metrics-release-2.0.0.csv"
data = pd.read_csv(file_path)

# Définir les features (X) et la cible (y)
# Remplacez 'target' par le nom exact de la colonne cible de vos données
X = data.drop(columns=['Bogue'])  # À adapter
y = data['Bogue']  # À adapter

# Filtrer les colonnes numériques uniquement
X_numeric = X.select_dtypes(include=[np.number])


# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Initialisation des modèles
log_reg = LogisticRegression(max_iter=200)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fonction pour appliquer le bootstrap et calculer les métriques
def bootstrap_metrics(X, y, model, n_iterations=100):
    metrics = {"auc": [], "recall": [], "precision": [], "f1": []}
    for i in range(n_iterations):
        # Créer un échantillon bootstrap
        X_resampled, y_resampled = resample(X, y, random_state=i, stratify=y)
        
        # Diviser les données bootstrap en train/test
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=i)
        
        # Entraîner le modèle
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calcul des métriques
        if y_pred_prob is not None:
            metrics["auc"].append(roc_auc_score(y_test, y_pred_prob))
        metrics["recall"].append(recall_score(y_test, y_pred, average="weighted"))
        metrics["precision"].append(precision_score(y_test, y_pred, average="weighted"))
        metrics["f1"].append(f1_score(y_test, y_pred, average="weighted"))
    
    return metrics

# Validation croisée en 10-fold et calcul des métriques
def cross_val_metrics(X, y, model):
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
    y_pred_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calcul des métriques
    auc = roc_auc_score(y, y_pred_prob) if y_pred_prob is not None else None
    recall = recall_score(y, y_pred, average="weighted")
    precision = precision_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")
    
    return {"auc": auc, "recall": recall, "precision": precision, "f1": f1}

# Calcul des métriques pour les deux approches et les deux modèles
metrics_log_reg_cv = cross_val_metrics(X_scaled, y, log_reg)
metrics_rf_cv = cross_val_metrics(X_scaled, y, rf_clf)

metrics_log_reg_bootstrap = bootstrap_metrics(X_scaled, y, log_reg)
metrics_rf_bootstrap = bootstrap_metrics(X_scaled, y, rf_clf)

# Affichage des résultats
print("Logistic Regression - Cross Validation Metrics:")
print(metrics_log_reg_cv)

print("\nRandom Forest - Cross Validation Metrics:")
print(metrics_rf_cv)

print("\nLogistic Regression - Bootstrap Metrics (Mean):")
print({k: np.mean(v) for k, v in metrics_log_reg_bootstrap.items()})

print("\nRandom Forest - Bootstrap Metrics (Mean):")
print({k: np.mean(v) for k, v in metrics_rf_bootstrap.items()})

# Visualisation des résultats Bootstrap
plt.figure(figsize=(12, 8))
for model_metrics, label, color in zip(
    [metrics_log_reg_bootstrap, metrics_rf_bootstrap],
    ['Logistic Regression', 'Random Forest'],
    ['blue', 'green']
):
    sns.histplot(model_metrics["auc"], color=color, label=f'{label} AUC', kde=True, stat="density")
    sns.histplot(model_metrics["f1"], color=color, label=f'{label} F1', kde=True, alpha=0.5, stat="density")

plt.title('Bootstrap Metrics Distribution - AUC and F1')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend()
plt.show()
