import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 📥 Charger les données
df = pd.read_csv(r"C:\Users\belgas\Desktop\PFA\creditcard.csv\creditcard.csv")

# 2. 🧹 Supprimer la colonne 'Time' si elle existe
if "Time" in df.columns:
    df = df.drop(columns=["Time"])

# 3. 🔀 Séparer le dataset en deux moitiés : 50% pour entraînement, 50% pour test
df_train, df_test = train_test_split(df, test_size=0.5, stratify=df["Class"], random_state=42)

# 4. 💾 Sauvegarder la moitié test dans un fichier CSV
test_file_path = r"C:\Users\belgas\Desktop\PFA\creditcard_test.csv"
df_test.to_csv(test_file_path, index=False)
print(f"✅ Fichier de test sauvegardé : {os.path.basename(test_file_path)}")

# 5. 🧪 Préparation des données d'entraînement
X_train = df_train.drop("Class", axis=1)
y_train = df_train["Class"]

# 6. 📏 Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 7. ⚖️ Équilibrage avec SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# 8. 🌳 Entraîner le modèle Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# 9. 💾 Sauvegarder le modèle et le scaler
joblib.dump(rf_model, r"C:\Users\belgas\Desktop\PFA\random_forest_model.pkl")
joblib.dump(scaler, r"C:\Users\belgas\Desktop\PFA\scaler.pkl")
print("✅ Modèle et scaler sauvegardés.")

# 🔍 Évaluation rapide sur l'entraînement
rf_predictions = rf_model.predict(X_train_scaled)

conf_mat = confusion_matrix(y_train, rf_predictions)
print("\n=== 🧪 Résultats internes (sur données d'entraînement) ===")
print(classification_report(y_train, rf_predictions))

# 🔷 Matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
plt.title("Matrice de confusion - Entraînement")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# 📈 Courbe ROC
y_train_proba = rf_model.predict_proba(X_train_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_train, y_train_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Entraînement")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n=== 🧪 Résultats internes ")

