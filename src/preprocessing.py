import re
import string
import spacy
# Il faut installer cette librairie : pip install langdetect
from langdetect import detect, LangDetectException

# 1. Chargement des DEUX modèles (Français et Anglais)
# On utilise des variables globales pour les charger une seule fois
try:
    nlp_fr = spacy.load("fr_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")
    print("✅ Succès : Modèles spaCy (FR + EN) chargés.")
except OSError:
    print("⚠️ ERREUR : Il manque un modèle spaCy.")
    print("Assure-toi d'avoir lancé :")
    print("python -m spacy download fr_core_news_sm")
    print("python -m spacy download en_core_web_sm")
    # On crée des variables vides pour éviter que le code crashe tout de suite
    nlp_fr = None
    nlp_en = None

def detect_language(text):
    """
    Détecte la langue du texte (fr ou en).
    Retourne 'fr' par défaut si le texte est trop court ou bizarre.
    """
    try:
        if not isinstance(text, str) or len(text) < 3:
            return "fr"
        return detect(text)
    except LangDetectException:
        return "fr"
    except Exception:
        return "fr"

def clean_text(text):
    """
    Pipeline bilingue :
    1. Détecte la langue
    2. Sélectionne le bon modèle spaCy (nlp_fr ou nlp_en)
    3. Nettoie (Tokenisation -> Stop-words -> Lemmatisation)
    """
    # Sécurité : si ce n'est pas du texte
    if not isinstance(text, str) or len(text) < 2:
        return ""

    # A. Détection de la langue pour choisir le bon modèle
    lang = detect_language(text)
    
    # B. Sélection du modèle (Routing)
    # On choisit le modèle anglais SI la langue est 'en' ET que le modèle est bien chargé
    if lang == 'en' and nlp_en is not None:
        nlp = nlp_en
    elif nlp_fr is not None:
        nlp = nlp_fr
    else:
        # Si aucun modèle n'est chargé (problème installation), on fait juste un nettoyage basique
        return text.lower().strip()

    # C. Nettoyage Regex (Commun aux deux langues)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    # D. Traitement NLP avec spaCy
    try:
        doc = nlp(text)
        tokens_propres = []
        for token in doc:
            # On garde le mot si ce n'est pas un stop-word, pas de la ponctuation, et > 1 lettre
            if not token.is_stop and not token.is_punct and len(token.text) > 1:
                tokens_propres.append(token.lemma_)
        
        return " ".join(tokens_propres)
    except Exception as e:
        print(f"Erreur NLP sur le texte : {e}")
        return text

# --- ZONE DE TEST ---
if __name__ == "__main__":
    print("--- Test du module Bilingue ---")
    
    phrase_fr = "Les clients n'ont pas aimé les produits livrés hier !"
    print(f"FR Original : {phrase_fr}")
    print(f"FR Nettoyé  : {clean_text(phrase_fr)}")
    
    print("-" * 20)
    
    phrase_en = "The delivery was very slow and I hate this product."
    print(f"EN Original : {phrase_en}")
    print(f"EN Nettoyé  : {clean_text(phrase_en)}")