from transformers import pipeline

# 1. Chargement du Pipeline (La fameuse "Boîte Noire")
# On spécifie un modèle "multilingue" capable de lire FR et EN
# Le téléchargement du modèle (environ 500Mo) se fera AUTOMATIQUEMENT à la première exécution.
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

try:
    # On crée l'analyseur
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    print(f"✅ Modèle Sentiment chargé : {model_name}")
except Exception as e:
    print(f"❌ Erreur chargement modèle : {e}")
    sentiment_pipeline = None

def analyze_sentiment(text):
    """
    Analyse le sentiment d'un texte.
    Retourne : (Label, Score, Couleur)
    Exemple : ("Positif", 0.95, "green")
    """
    if not text or not isinstance(text, str) or sentiment_pipeline is None:
        return "Neutre", 0.0, "gray"

    # Le pipeline n'aime pas les textes trop longs (> 512 mots). On coupe si besoin.
    text = text[:512]

    try:
        # APPEL MAGIQUE : On donne le texte à l'IA
        result = sentiment_pipeline(text)[0]
        # result ressemble à : {'label': '5 stars', 'score': 0.85}
        
        label_brut = result['label'] # ex: "1 star", "4 stars"
        score = result['score']      # Confiance de l'IA (0.0 à 1.0)
        
        # On convertit les étoiles en sentiments humains
        # Le modèle renvoie '1 star', '2 stars', etc.
        star_rating = int(label_brut.split()[0]) # On récupère le chiffre 1, 2...

        if star_rating <= 2:
            return "Négatif 😡", score, "red"
        elif star_rating == 3:
            return "Neutre 😐", score, "orange"
        else:
            return "Positif 😃", score, "green"

    except Exception as e:
        print(f"Erreur analyse : {e}")
        return "Erreur", 0.0, "black"

# --- TEST RAPIDE ---
if __name__ == "__main__":
    print("--- Test du module Sentiment ---")
    print("⏳ Premier lancement : Le modèle va se télécharger (patience...)...")
    
    avis_1 = "Ce produit est nul, je le déteste."
    avis_2 = "I absolutely love this ! Best purchase ever."
    
    print(f"Avis FR : {avis_1} -> {analyze_sentiment(avis_1)}")
    print(f"Avis EN : {avis_2} -> {analyze_sentiment(avis_2)}")