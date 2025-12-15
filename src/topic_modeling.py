from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np

def run_topic_modeling(df, text_column='Avis_Nettoye', n_topics=3):
    """
    Exécute le Topic Modeling et retourne :
    1. Le DataFrame enrichi avec une colonne 'Sujet_Dominant'
    2. Un dictionnaire des mots-clés pour l'affichage
    """
    
    # 1. VECTORISATION
    # On évite les mots trop fréquents (95%) et trop rares
    vectorizer = CountVectorizer(max_df=0.95, min_df=1)
    try:
        dtm = vectorizer.fit_transform(df[text_column])
    except ValueError:
        # Si le vocabulaire est vide (ex: textes vides)
        return df, {}

    # 2. MODÉLISATION LDA
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(dtm)

    # 3. ATTRIBUTION DES SUJETS (C'est l'étape qui manquait !)
    # transform retourne la probabilité d'appartenance [0.1, 0.9] -> Sujet 2 gagne
    topic_results = lda_model.transform(dtm)
    
    # argmax récupère l'index du sujet le plus fort (0, 1, 2...)
    df['Topic_ID'] = topic_results.argmax(axis=1)
    
    # On crée un nom plus joli "Sujet 1", "Sujet 2"
    df['Sujet_Dominant'] = df['Topic_ID'].apply(lambda x: f"Sujet {x + 1}")

    # 4. EXTRACTION DES MOTS CLÉS (Pour l'affichage)
    topics_keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for index, topic in enumerate(lda_model.components_):
        top_words_indices = topic.argsort()[-10:]
        top_words = [feature_names[i] for i in top_words_indices]
        topics_keywords[f"Sujet {index + 1}"] = ", ".join(top_words)
        
    return df, topics_keywords