import streamlit as st
import pandas as pd
from src.database import SessionLocal, Review
# On importe notre fonction intelligente
from src.preprocessing import clean_text 

st.set_page_config(page_title="Analyse Avis Clients", layout="wide")
st.title("📊 Analyse d'Avis Clients - NLP Processor")

# --- FONCTION DE SAUVEGARDE (Identique à avant) ---
def save_to_db(df, text_column, source_column='source'):
    session = SessionLocal()
    count = 0
    try:
        for index, row in df.iterrows():
            source_val = row[source_column] if source_column in df.columns else "upload_manuel"
            
            # NOUVEAU : On sauvegarde aussi la version nettoyée !
            # Note : Pour l'instant notre BDD n'a pas de colonne dédiée "cleaned_text", 
            # on va sauvegarder le brut pour l'instant, le nettoyage se fera à la volée ou 
            # on modifiera la BDD plus tard. Restons simples pour l'instant.
            new_review = Review(
                text_content=row[text_column],
                source=source_val
            )
            session.add(new_review)
            count += 1
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        st.error(f"Erreur sauvegarde : {e}")
        return 0
    finally:
        session.close()

# --- INTERFACE ---
uploaded_file = st.file_uploader("Choisissez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    colonnes_possibles = ['commentaire', 'avis', 'review', 'text']
    col_trouvee = [c for c in df.columns if c.lower() in colonnes_possibles]
    
    if col_trouvee:
        col_texte = col_trouvee[0]
        
        st.info("Traitement NLP en cours... (Tokenisation & Lemmatisation)")
        
        # --- PARTIE NLP ---
        # On applique le nettoyage spaCy
        # On utilise une barre de progression car ça peut prendre quelques secondes
        barre = st.progress(0)
        
        # On applique la fonction ligne par ligne
        # Astuce dev : on pourrait optimiser, mais pour <1000 lignes c'est ok
        df['Avis_Nettoye'] = df[col_texte].apply(clean_text)
        barre.progress(100)
        
        st.success("Traitement NLP terminé !")
        
        # --- COMPARAISON VISUELLE ---
        st.subheader("🧐 Comparaison : Brut vs Lemmatisé")
        st.write("Regarde comment l'IA simplifie les phrases pour n'en garder que le sens :")
        
        # On affiche 2 colonnes pour comparer
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Texte Original**")
            st.dataframe(df[[col_texte]].head(10), use_container_width=True)
        with col2:
            st.markdown("**Texte Nettoyé (Lemmes)**")
            st.dataframe(df[['Avis_Nettoye']].head(10), use_container_width=True)
            
        # Bouton sauvegarde (inchangé)
        if st.button("💾 Sauvegarder (Brut) dans la BDD"):
            nb = save_to_db(df, col_texte)
            if nb > 0: st.success(f"{nb} avis sauvegardés.")

    else:
        st.error("Colonne texte manquante.")