import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 📥 Lire le dataset
df = pd.read_csv(r"C:\Users\belgas\Desktop\PFA\creditcard.csv\creditcard.csv")

# 🔎 Afficher infos de base
print("Dimensions:", df.shape)
print("Valeurs nulles par colonne:")
print(df.isnull().sum())

# 📊 Standardiser la colonne Amount
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# 🗑️ Supprimer la colonne Time (pas utile ici)
df = df.drop(columns=['Time'])

# 🎯 Séparer features (X) et label (y)
X = df.drop('Class', axis=1)
y = df['Class']

# ⚖️ Appliquer SMOTE pour équilibrer
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# 📦 Sauvegarder les données prétraitées
processed_df = pd.concat([X_res, y_res], axis=1)
processed_df.to_csv("processed_data.csv", index=False)
print("\n✅ Données prétraitées sauvegardées sous 'processed_data.csv'")

# ✅ Résultat
print("\nAprès SMOTE:")
print("Nouvelle forme des données :", X_res.shape)
print("Répartition des classes :")
print(pd.Series(y_res).value_counts())