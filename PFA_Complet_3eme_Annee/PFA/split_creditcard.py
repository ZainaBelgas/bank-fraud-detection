import pandas as pd

# Charger tout le fichier
df = pd.read_csv(r"C:\Users\belgas\Desktop\PFA\creditcard.csv\creditcard.csv")

# Mélanger les données (important pour que ce soit aléatoire)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Diviser en deux moitiés
half = len(df) // 2
df_train = df.iloc[:half]
df_test = df.iloc[half:]

# Sauvegarder en deux fichiers CSV
df_train.to_csv("creditcard_train.csv", index=False)
df_test.to_csv("creditcard_test.csv", index=False)

print("Fichiers créés : creditcard_train.csv et creditcard_test.csv")
