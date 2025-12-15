import re
import string
import spacy

# 1. Chargement du modèle français
# On le charge une seule fois au début du script pour la performance
# "sm" = Small (rapide)
try:
    nlp = spacy.load("fr_core_news_sm")
    # On ajoute des mots vides personnalisés si besoin (ex: "client", "avis"...)
    # nlp.Defaults.stop_words.add("ceci_est_un_test")
except OSError:
    print("Erreur : Le modèle spaCy n'est pas trouvé. As-tu lancé 'python -m spacy download fr_core_news_sm' ?")

def clean_text(text):
    """
    Pipeline complet de nettoyage NLP :
    1. Nettoyage Regex (bruit)
    2. NLP (Tokenisation, Stop-words, Lemmatisation)
    """
    if not isinstance(text, str):
        return ""

    # --- ÉTAPE 1 : Nettoyage de surface (Regex) ---
    text = text.lower() # Minuscules
    text = re.sub(r'\s+', ' ', text).strip() # Espaces en trop

    # --- ÉTAPE 2 : Traitement NLP intelligent avec spaCy ---
    
    # On nourrit le modèle avec le texte
    doc = nlp(text)
    
    tokens_propres = []
    
    # On parcourt chaque mot (token) identifié par l'IA
    for token in doc:
        # Conditions pour GARDER un mot :
        # 1. Pas un stop-word (le, la, un, du...)
        # 2. Pas de la ponctuation (!, ?, .)
        # 3. Plus long que 1 lettre (pour éviter les erreurs)
        if not token.is_stop and not token.is_punct and len(token.text) > 1:
            # On prend le LEMME (la racine du mot)
            # ex: "aimaient" devient "aimer"
            tokens_propres.append(token.lemma_)
    
    # On reconstruit le texte propre (rejoint par des espaces)
    return " ".join(tokens_propres)

# --- ZONE DE TEST ---
if __name__ == "__main__":
    # Phrase complexe pour tester
    phrase_test = "Les clients n'ont pas aimé les produits livrés hier !"
    
    print(f"Phrase originale : {phrase_test}")
    print(f"Après nettoyage  : {clean_text(phrase_test)}")