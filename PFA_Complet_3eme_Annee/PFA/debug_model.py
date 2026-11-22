"""
Debug script to understand why the model isn't detecting fraud
Save this as 'debug_model.py' and run it
"""

import joblib
import pandas as pd
import numpy as np

def debug_model():
    print("🔍 DEBUG: Analyse détaillée du modèle")
    print("="*60)
    
    try:
        # 1. Charger le modèle et le scaler
        print("📥 Chargement du modèle...")
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        print("✅ Modèle et scaler chargés")
        
        # 2. Informations sur le modèle
        print(f"\n📊 Informations du modèle:")
        print(f"Type: {type(model)}")
        print(f"Nombre de features attendues: {model.n_features_in_}")
        print(f"Classes: {model.classes_}")
        
        # 3. Informations sur le scaler
        print(f"\n📏 Informations du scaler:")
        print(f"Type: {type(scaler)}")
        print(f"Nombre de features: {len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 'N/A'}")
        print(f"Shape des moyennes: {scaler.mean_.shape if hasattr(scaler, 'mean_') else 'N/A'}")
        
        # 4. Test avec les données suspectes
        print(f"\n🚨 Test avec transaction suspecte:")
        suspicious_data = {
            'Amount': 50000,
            'V1': 15, 'V2': -8.5, 'V3': 25, 'V4': -12, 'V5': 18
        }
        
        debug_prediction(model, scaler, suspicious_data, "Suspecte")
        
        # 5. Test avec les données normales
        print(f"\n✅ Test avec transaction normale:")
        normal_data = {
            'Amount': 149.62,
            'V1': -1.35, 'V2': -0.07, 'V3': 2.53, 'V4': 1.37, 'V5': -0.33
        }
        
        debug_prediction(model, scaler, normal_data, "Normale")
        
        # 6. Test avec des valeurs encore plus extrêmes
        print(f"\n⚠️ Test avec valeurs extrêmes:")
        extreme_data = {
            'Amount': 999999,
            'V1': 999, 'V2': -999, 'V3': 999, 'V4': -999, 'V5': 999
        }
        
        debug_prediction(model, scaler, extreme_data, "Extrême")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def debug_prediction(model, scaler, data, label):
    """Fonction de debug détaillée pour une prédiction"""
    
    print(f"\n--- Debug pour transaction {label} ---")
    print(f"Données d'entrée: {data}")
    
    # Créer le vecteur de features
    features = [0.0] * 29
    features[0] = data['V1']
    features[1] = data['V2'] 
    features[2] = data['V3']
    features[3] = data['V4']
    features[4] = data['V5']
    features[28] = data['Amount']
    
    print(f"Vecteur features (premiers 10): {features[:10]}")
    print(f"Amount à l'index 28: {features[28]}")
    
    # Créer DataFrame
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    input_df = pd.DataFrame([features], columns=feature_names)
    
    print(f"Shape du DataFrame: {input_df.shape}")
    print(f"Colonnes: {list(input_df.columns)}")
    
    # Avant scaling
    print(f"Avant scaling - Amount: {input_df['Amount'].iloc[0]}")
    print(f"Avant scaling - V1: {input_df['V1'].iloc[0]}")
    
    # Appliquer le scaler
    input_scaled = scaler.transform(input_df)
    
    print(f"Après scaling - Shape: {input_scaled.shape}")
    print(f"Après scaling - premiers 5 valeurs: {input_scaled[0][:5]}")
    print(f"Après scaling - dernière valeur (Amount): {input_scaled[0][-1]}")
    
    # Prédiction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    print(f"Prédiction: {prediction}")
    print(f"Probabilités [Normal, Fraude]: {probabilities}")
    print(f"Probabilité de fraude: {probabilities[1]:.4f}")
    
    # Diagnostic
    if probabilities[1] < 0.1:
        print("⚠️ PROBLÈME: Probabilité de fraude très faible")
    elif probabilities[1] > 0.9:
        print("✅ Forte probabilité de fraude détectée")
    else:
        print("🔶 Probabilité modérée")
    
    return prediction, probabilities

if __name__ == "__main__":
    debug_model()
    input("\nAppuyez sur Entrée pour fermer...")