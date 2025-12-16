import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
# Correction de l'import : on précise 'src.sentiment' pour que ça marche depuis le dashboard
from src.sentiment import analyze_sentiment

# 1. CRÉATION DU JEU DE DONNÉES "VÉRITÉ TERRAIN"
donnees_test = [
    {"text": "J'adore ce produit, il est génial !", "verite": "Positif 😃"},
    {"text": "C'est une catastrophe, je déteste.", "verite": "Négatif 😡"},
    {"text": "Livraison rapide et soignée.", "verite": "Positif 😃"},
    {"text": "Bof, ça passe mais c'est cher.", "verite": "Neutre 😐"},
    {"text": "Le service client ne répond jamais.", "verite": "Négatif 😡"},
    {"text": "Correct sans plus.", "verite": "Neutre 😐"},
    {"text": "Best purchase ever, I love it!", "verite": "Positif 😃"},
    {"text": "Very bad quality.", "verite": "Négatif 😡"}
]

# 2. FONCTION APPELÉE PAR LE DASHBOARD
def get_metrics():
    """
    Fonction qui calcule l'Accuracy et le F1-Score.
    Retourne : (accuracy, f1_score, nombre_echantillons)
    """
    y_true = []
    y_pred = []
    
    # On boucle sur chaque phrase test
    for item in donnees_test:
        text = item["text"]
        realite = item["verite"]
        
        # On demande à l'IA
        pred_label, score, _ = analyze_sentiment(text)
        
        y_true.append(realite)
        y_pred.append(pred_label)

    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted') 
    
    return accuracy, f1, len(donnees_test)

# Petit bloc pour tester ce fichier tout seul si besoin
if __name__ == "__main__":
    acc, f1, n = get_metrics()
    print(f"Test manuel : Accuracy={acc}, F1={f1}")