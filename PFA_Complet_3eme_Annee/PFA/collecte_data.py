# collecte_data.py
import pandas as pd
# Lecture du fichier CSV contenant les données de transaction
df = pd.read_csv(r"C:\Users\belgas\Desktop\PFA\creditcard.csv\creditcard.csv")

print(df.head())
  # Assure-toi que le fichier est bien placé dans le dossier 'data'
# Affichage de la taille du dataset (lignes, colonnes)
print("Dimensions du dataset :", df.shape)
# Aperçu des 5 premières lignes
print("\nAperçu du dataset :")
print(df.head())
# Analyse de la répartition des classes (0 = normal, 1 = fraude)
print("\nRépartition des classes (0 = normal, 1 = fraude) :")
print(df['Class'].value_counts())

print(df.isnull().sum())  # Vérifie les valeurs manquantes
                            