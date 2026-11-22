from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "votre_cle_secrete"  # 🔐 Pour les sessions

#  MODÈLE INTÉGRÉ - Plus besoin de fichiers .pkl !
class ModeleDetectionFraude:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.entrainer_modele()
    
    def entrainer_modele(self):
        """Créer et entraîner le modèle directement dans l'app"""
        print("🔧 Initialisation du modèle de détection de fraude...")
        
        # Créer des données d'entraînement avec patterns clairs
        donnees_entrainement = self.creer_donnees_entrainement()
        
        # Séparer X et y
        X = donnees_entrainement.drop('Class', axis=1)
        y = donnees_entrainement['Class']
        
        # Entraîner le scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraîner le modèle
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        
        print("✅ Modèle de fraude initialisé et prêt !")
    
    def creer_donnees_entrainement(self):
        """Créer des données synthétiques avec patterns de fraude clairs"""
        
        # Transactions normales (80%)
        n_normal = 4000
        donnees_normales = []
        
        for i in range(n_normal):
            v1 = np.random.normal(0, 1.5)
            v2 = np.random.normal(0, 1.5) 
            v3 = np.random.normal(0, 1.5)
            v4 = np.random.normal(0, 1.5)
            v5 = np.random.normal(0, 1.5)
            montant = np.random.lognormal(4, 1)  # 20-500€
            
            # V6-V28 avec petites valeurs
            autres_vs = np.random.normal(0, 1, 23)
            
            ligne = [v1, v2, v3, v4, v5] + list(autres_vs) + [montant, 0]
            donnees_normales.append(ligne)
        
        # Transactions frauduleuses (20%)
        n_fraude = 1000
        donnees_fraude = []
        
        for i in range(n_fraude):
            if i < n_fraude // 2:
                # Pattern 1: Valeurs V élevées
                v1 = np.random.uniform(5, 20)
                v2 = np.random.uniform(-20, -5)
                v3 = np.random.uniform(10, 30)
                v4 = np.random.uniform(-15, -5)
                v5 = np.random.uniform(8, 25)
                montant = np.random.uniform(5000, 100000)
            else:
                # Pattern 2: Valeurs négatives extrêmes
                v1 = np.random.uniform(-25, -8)
                v2 = np.random.uniform(10, 25)
                v3 = np.random.uniform(-30, -10)
                v4 = np.random.uniform(12, 20)
                v5 = np.random.uniform(-20, -8)
                montant = np.random.uniform(10000, 200000)
            
            autres_vs = np.random.normal(0, 3, 23)
            ligne = [v1, v2, v3, v4, v5] + list(autres_vs) + [montant, 1]
            donnees_fraude.append(ligne)
        
        # Combiner et créer DataFrame
        toutes_donnees = donnees_normales + donnees_fraude
        colonnes = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        df = pd.DataFrame(toutes_donnees, columns=colonnes)
        
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def predire(self, v1, v2, v3, v4, v5, montant):
        """Prédire si une transaction est frauduleuse"""
        
        # Créer le vecteur de features
        features = [0.0] * 29
        features[0] = v1
        features[1] = v2
        features[2] = v3
        features[3] = v4
        features[4] = v5
        features[28] = montant
        
        # Créer DataFrame
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Normaliser et prédire
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)[0]
        probabilites = self.model.predict_proba(input_scaled)[0]
        
        return prediction, probabilites

# Initialiser le modèle au démarrage de l'app
detecteur_fraude = ModeleDetectionFraude()

# Fichier de logs
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs.json")

# S'assurer que logs.json existe
if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# === ROUTE DE CONNEXION ===
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "1234":
            session["user"] = username
            flash("Connexion réussie ✅", "success")
            return redirect(url_for("index"))
        else:
            flash("Identifiants incorrects ❌", "danger")

    return render_template("login.html")

# === ROUTE DE DECONNEXION ===
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Déconnexion réussie 👋", "info")
    return redirect(url_for("login"))

# === PAGE PRINCIPALE ===
@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            # Récupération des données du formulaire
            amount = float(request.form['Amount'])
            v1 = float(request.form['V1'])
            v2 = float(request.form['V2'])
            v3 = float(request.form['V3'])
            v4 = float(request.form['V4'])
            v5 = float(request.form['V5'])

            # Utiliser le modèle intégré pour la prédiction
            prediction, probabilites = detecteur_fraude.predire(v1, v2, v3, v4, v5, amount)
            
            # Résultat basé sur la prédiction
            if prediction == 1:
                result = f"🚨 Fraude détectée ! (Probabilité: {probabilites[1]:.1%})"
            else:
                result = f"✅ Transaction normale (Risque de fraude: {probabilites[1]:.1%})"

            # Enregistrement dans les logs
            log_entry = {
                "Amount": amount,
                "V1": v1,
                "V2": v2,
                "V3": v3,
                "V4": v4,
                "V5": v5,
                "Prediction": int(prediction),
                "Probabilite_Fraude": float(probabilites[1])
            }

            try:
                with open(LOG_FILE, "r+") as f:
                    data = json.load(f)
                    data.append(log_entry)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            except json.JSONDecodeError:
                with open(LOG_FILE, "w") as f:
                    json.dump([log_entry], f, indent=4)

            return redirect(url_for("result", prediction=result))

        except Exception as e:
            flash(f"Erreur dans la prédiction: {e}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html", result=None, user=session.get("user"))

# === PAGE RESULTAT ===
@app.route("/result")
def result():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = request.args.get("prediction", "Résultat inconnu")
    return render_template("result.html", prediction=prediction, user=session.get("user"))

# === PAGE DES STATISTIQUES ===
@app.route("/stats")
def stats():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        data = []

    total = len(data)
    total_fraude = sum(entry['Prediction'] for entry in data)
    taux_fraude = (total_fraude / total * 100) if total > 0 else 0

    return render_template(
        "stats.html",
        total=total,
        total_fraude=total_fraude,
        taux_fraude=taux_fraude,
        logs=data,
        user=session.get("user")
    )

# === TEST DE L'API (optionnel) ===
@app.route("/test_modele")
def test_modele():
    """Route pour tester rapidement le modèle"""
    if "user" not in session:
        return redirect(url_for("login"))
    
    # Test avec transaction suspecte
    prediction, probabilites = detecteur_fraude.predire(15, -8.5, 25, -12, 18, 50000)
    
    resultat = {
        "transaction_test": "Valeurs suspectes (15, -8.5, 25, -12, 18, 50000)",
        "prediction": int(prediction),
        "probabilites": {
            "normal": float(probabilites[0]),
            "fraude": float(probabilites[1])
        },
        "interpretation": "🚨 FRAUDE" if prediction == 1 else "✅ NORMAL"
    }
    
    return json.dumps(resultat, indent=2)

# === LANCEMENT ===
if __name__ == "__main__":
    print("🚀 Démarrage de l'application de détection de fraude...")
    print("📊 Modèle intégré - Aucun fichier .pkl requis !")
    app.run(debug=True)
