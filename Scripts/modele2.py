import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data = pd.read_csv(r"C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Clean_Metrics\result-metrics-release-2.0.0.csv")

# Pré-traitement
data = data.replace({',': '.'}, regex=True)

# Assurez-vous que toutes les colonnes numériques sont de type float
for column in data.select_dtypes(include='object').columns:
    try:
        data[column] = data[column].astype(float)
    except ValueError:
        pass  # Ignore les colonnes non convertibles (comme celles avec des chaînes non numériques)

# Vérifier les types après conversion
print(data.dtypes)
data['Bogue'] = data['Bogue'].astype(int)

# Sélectionner les caractéristiques (toutes les colonnes sauf 'Bogue', 'Kind', 'Name', 'Version', 'CommitId', 'BaseFileName')
X = data.drop(columns=['Bogue', 'Kind', 'Name', 'Version', 'CommitId', 'BaseFileName'])

# Variable cible
y = data['Bogue']

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialiser les modèles
log_reg_model = LogisticRegression(max_iter=1000, solver='lbfgs')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fonction pour calculer les métriques
def compute_metrics(model, X_test, y_test):
    predicted_scores = model.predict_proba(X_test)[:, 1]
    predicted_labels = (predicted_scores > 0.5).astype(int)
    
    auc_score = roc_auc_score(y_test, predicted_scores)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    
    return auc_score, precision, recall, predicted_labels

# Nombre d'itérations pour le bootstrap
n_bootstraps = 100

# Listes pour stocker les scores
log_reg_AUC = []
log_reg_Precision = []
log_reg_Recall = []

rf_AUC = []
rf_Precision = []
rf_Recall = []

# Listes pour stocker les importances des variables
log_reg_importances = []
rf_importances = []

# Boucle pour les échantillons bootstrap
for i in range(n_bootstraps):
    # Créer un échantillon bootstrap
    X_train, y_train = resample(X_scaled, y, random_state=i)
    
    # Créer le jeu de test en utilisant train_test_split pour gérer l'alignement des indices
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=i)
    
    # Entraîner et évaluer le modèle de régression logistique
    log_reg_model.fit(X_train, y_train)
    log_reg_auc, log_reg_precision, log_reg_recall, _ = compute_metrics(log_reg_model, X_test, y_test)
    log_reg_AUC.append(log_reg_auc)
    log_reg_Precision.append(log_reg_precision)
    log_reg_Recall.append(log_reg_recall)
    
    # Entraîner et évaluer le modèle Random Forest
    rf_model.fit(X_train, y_train)
    rf_auc, rf_precision, rf_recall, _ = compute_metrics(rf_model, X_test, y_test)
    rf_AUC.append(rf_auc)
    rf_Precision.append(rf_precision)
    rf_Recall.append(rf_recall)
    
    # Stocker les importances des variables
    log_reg_importances.append(np.abs(log_reg_model.coef_[0]))  # Importances par coefficient
    rf_importances.append(rf_model.feature_importances_)  # Importances pour RandomForest

# Calculer les moyennes et les écarts-types des résultats
log_reg_auc_mean = np.mean(log_reg_AUC)
log_reg_precision_mean = np.mean(log_reg_Precision)
log_reg_recall_mean = np.mean(log_reg_Recall)

rf_auc_mean = np.mean(rf_AUC)
rf_precision_mean = np.mean(rf_Precision)
rf_recall_mean = np.mean(rf_Recall)

log_reg_auc_std = np.std(log_reg_AUC)
log_reg_precision_std = np.std(log_reg_Precision)
log_reg_recall_std = np.std(log_reg_Recall)

rf_auc_std = np.std(rf_AUC)
rf_precision_std = np.std(rf_Precision)
rf_recall_std = np.std(rf_Recall)

# Afficher les résultats
print("Logistic Regression - AUC: {:.4f} ± {:.4f}, Precision: {:.4f} ± {:.4f}, Recall: {:.4f} ± {:.4f}".format(
    log_reg_auc_mean, log_reg_auc_std, log_reg_precision_mean, log_reg_precision_std, log_reg_recall_mean, log_reg_recall_std))

print("Random Forest - AUC: {:.4f} ± {:.4f}, Precision: {:.4f} ± {:.4f}, Recall: {:.4f} ± {:.4f}".format(
    rf_auc_mean, rf_auc_std, rf_precision_mean, rf_precision_std, rf_recall_mean, rf_recall_std))

# Matrice de confusion pour Logistic Regression
cm_log_reg = confusion_matrix(y_test, _)
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.title("Matrice de Confusion - Logistic Regression")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.show()

# Matrice de confusion pour Random Forest
cm_rf = confusion_matrix(y_test, _)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.title("Matrice de Confusion - Random Forest")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.show()

# Calculer les moyennes des importances des variables
mean_log_reg_importances = np.mean(log_reg_importances, axis=0)
mean_rf_importances = np.mean(rf_importances, axis=0)

# Création du DataFrame pour les résultats
log_reg_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_log_reg_importances
}).sort_values(by='Importance', ascending=False)

rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_rf_importances
}).sort_values(by='Importance', ascending=False)

# Visualisation de l'importance des variables pour Logistic Regression
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=log_reg_importance_df.head(10), palette='viridis')
plt.title("Top 10 Features - Logistic Regression")
plt.show()

# Visualisation de l'importance des variables pour Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance_df.head(10), palette='viridis')
plt.title("Top 10 Features - Random Forest")
plt.show()
