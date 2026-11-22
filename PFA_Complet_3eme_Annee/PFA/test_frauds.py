import joblib
import pandas as pd

# 1. ⏬ Charger le modèle et le scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# 2. 📄 Charger les données de test (la moitié sauvegardée dans model_training.py)
df = pd.read_csv(r"C:\Users\belgas\Desktop\PFA\creditcard_test.csv")

# 3. 🧹 Supprimer la colonne 'Time' si elle existe
if "Time" in df.columns:
    df = df.drop(columns=["Time"])

# 4. 🏷️ Séparer les données en X et y
X_test = df.drop("Class", axis=1)
y_test = df["Class"]

# 5. 📏 Appliquer le scaler sur toutes les features (car dans model_training.py on a standardisé tout)
X_test_scaled = scaler.transform(X_test)

# 6. 🤖 Prédictions du modèle
predictions = model.predict(X_test_scaled)

# 7. 📊 Ajouter les résultats au DataFrame
df["Predicted"] = predictions
df["Predicted_Label"] = df["Predicted"].map({0: "Normal", 1: "Fraude"})
df["Correct"] = df["Class"] == df["Predicted"]

# 8. ✅ Afficher les 20 premiers résultats
print("\n===== 🧪 Résultats de prédiction =====")
print(df[["Amount", "Class", "Predicted", "Predicted_Label", "Correct"]].head(20))

# 9. 🧮 Calcul de la précision globale
accuracy = (df["Correct"].sum() / len(df)) * 100
print(f"\n🎯 Précision totale sur les données test : {accuracy:.2f}%")
