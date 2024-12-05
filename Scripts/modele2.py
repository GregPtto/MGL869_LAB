import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

results_folder = "model_results_3.0.0"
os.makedirs(results_folder, exist_ok=True)

data = pd.read_csv(r'C:\Users\gregs\Desktop\Canada\Cours\MGL869\MGL869_LAB\CSV_files\Clean_Metrics\result-metrics-rel-release-3.0.0.csv')

data = data.replace({',': '.'}, regex=True)

for column in data.select_dtypes(include='object').columns:
    try:
        data[column] = data[column].astype(float)
    except ValueError:
        pass

X = data.drop(columns=['Bogue', 'Kind', 'Name', 'Version', 'CommitId', 'BaseFileName'])

y = data['Bogue']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_reg_model = LogisticRegression(max_iter=1000, solver='lbfgs')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

def compute_metrics(model, X_test, y_test):
    predicted_scores = model.predict_proba(X_test)[:, 1]
    predicted_labels = (predicted_scores > 0.5).astype(int)
    
    auc_score = roc_auc_score(y_test, predicted_scores)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    
    return auc_score, precision, recall, predicted_labels

n_bootstraps = 100

log_reg_AUC = []
log_reg_Precision = []
log_reg_Recall = []

rf_AUC = []
rf_Precision = []
rf_Recall = []

log_reg_importances = []
rf_importances = []


for i in range(n_bootstraps):
    train_indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
    test_indices = np.setdiff1d(np.arange(len(X_scaled)), train_indices)
    
    X_train, y_train = X_scaled[train_indices], y.iloc[train_indices]
    X_test, y_test = X_scaled[test_indices], y.iloc[test_indices]
    
    log_reg_model.fit(X_train, y_train)
    log_reg_auc, log_reg_precision, log_reg_recall, log_reg_predictions = compute_metrics(log_reg_model, X_test, y_test)
    log_reg_AUC.append(log_reg_auc)
    log_reg_Precision.append(log_reg_precision)
    log_reg_Recall.append(log_reg_recall)
    
    rf_model.fit(X_train, y_train)
    rf_auc, rf_precision, rf_recall, rf_predictions = compute_metrics(rf_model, X_test, y_test)
    rf_AUC.append(rf_auc)
    rf_Precision.append(rf_precision)
    rf_Recall.append(rf_recall)
    
    log_reg_importances.append(np.abs(log_reg_model.coef_[0]))
    rf_importances.append(rf_model.feature_importances_)


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

dataTest = pd.DataFrame({"Model": ["Logistic Regression", "Random Forest"],
    "AUC": [log_reg_auc_mean, rf_auc_mean],
    "Precision": [log_reg_precision_mean, rf_precision_mean],
    "Recall": [log_reg_recall_mean, rf_recall_mean]})


dataTest.to_csv(os.path.join(results_folder,"perf.csv"), index=False)

cm_log_reg = confusion_matrix(y_test, log_reg_model.predict(X_test))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.title("Matrice de Confusion - Logistic Regression")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.savefig(os.path.join(results_folder, "cm_log_reg.png"))
plt.close()

cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.title("Matrice de Confusion - Random Forest")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.savefig(os.path.join(results_folder, "cm_rf.png"))

mean_log_reg_importances = np.mean(log_reg_importances, axis=0)
mean_rf_importances = np.mean(rf_importances, axis=0)

log_reg_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_log_reg_importances
}).sort_values(by='Importance', ascending=False)

rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_rf_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=log_reg_importance_df.head(10), palette='viridis')
plt.title("Top 10 Features - Logistic Regression")
plt.savefig(os.path.join(results_folder, "Logistic_regression_importances.png"))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance_df.head(10), palette='viridis')
plt.title("Top 10 Features - Random Forest")
plt.savefig(os.path.join(results_folder, "Random_forest_feature_importances.png"))

readme_path = os.path.join(results_folder, "README.md")
with open(readme_path, "w") as readme_file:
    readme_file.write("# Model Evaluation Results\n\n")
    readme_file.write("## Metrics\n")
    readme_file.write(dataTest.to_markdown(index=False))
    readme_file.write("\n\n## Visualizations\n")
    readme_file.write("### Matrice de Confusion\n")
    readme_file.write("|**Random Forest** | **Logistic Regression**|\n")
    readme_file.write(":-----------------:|:-----------------------:\n")
    readme_file.write("![Confusion Matrix](cm_rf.png) | ![Confusion Matrix](cm_log_reg.png)\n")
    readme_file.write("\n### Feature Importances\n")
    readme_file.write("**Random Forest**\n")
    readme_file.write(rf_importance_df.head(10).to_markdown(index=False))
    readme_file.write("\n---")
    readme_file.write("\n\n**Logistic Regression**\n")
    readme_file.write(log_reg_importance_df.head(10).to_markdown(index=False))
    readme_file.write("\n\n|**Random Forest** | **Logistic Regression**|\n")
    readme_file.write(":-----------------:|:-----------------------:\n")
    readme_file.write("![Feature Importances](Random_forest_feature_importances.png) | ![Feature Importances](Logistic_regression_importances.png)\n")
