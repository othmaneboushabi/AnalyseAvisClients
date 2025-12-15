import streamlit as st
import pandas as pd
# MODIFICATION 1 : On ajoute init_db ici 👇
from src.database import SessionLocal, Review, init_db 
from src.preprocessing import clean_text, detect_language
from src.topic_modeling import run_topic_modeling

st.set_page_config(page_title="Analyse Avis Clients", layout="wide")
st.title("📊 Analyse d'Avis Clients - Stockage Topic")

# MODIFICATION 2 : On lance la création des tables au démarrage 👇
init_db()

# --- FONCTION DE SAUVEGARDE MISE À JOUR ---
def save_to_db(df, text_column, source_column='source'):
    session = SessionLocal()
    count = 0
    try:
        for index, row in df.iterrows():
            source_val = row[source_column] if source_column in df.columns else "upload_manuel"
            
            # On vérifie si on a fait l'analyse de topic
            topic_val = row['Sujet_Dominant'] if 'Sujet_Dominant' in df.columns else None
            
            new_review = Review(
                text_content=row[text_column],
                source=source_val,
                topic=topic_val  # <--- ON SAUVEGARDE LE TOPIC ICI
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
    
    col_trouvee = [c for c in df.columns if c.lower() in ['commentaire', 'avis', 'review', 'text']]
    
    if col_trouvee:
        col_texte = col_trouvee[0]
        
        # 1. NLP
        with st.spinner('Traitement NLP...'):
            df['Langue'] = df[col_texte].apply(detect_language)
            df['Avis_Nettoye'] = df[col_texte].apply(clean_text)
        
        st.success("NLP terminé.")
        
        # 2. TOPIC MODELING (Nouvelle version)
        st.divider()
        st.subheader("🧠 Analyse Sémantique")
        
        try:
            # Cette fonction modifie df directement en ajoutant la colonne 'Sujet_Dominant'
            df, topics_display = run_topic_modeling(df, 'Avis_Nettoye', n_topics=2)
            
            # Affichage des mots clés
            cols = st.columns(2)
            for i, (sujet, mots) in enumerate(topics_display.items()):
                with cols[i % 2]:
                    st.info(f"**{sujet}** : {mots}")
            
            # Affichage du tableau avec la nouvelle colonne
            st.write("### Résultat de l'attribution :")
            st.dataframe(df[['Avis_Nettoye', 'Sujet_Dominant']].head(5))

        except Exception as e:
            st.warning(f"Erreur Topic Modeling : {e}")

        # 3. SAUVEGARDE
        if st.button("💾 Sauvegarder (Texte + Sujet) dans la BDD"):
            nb = save_to_db(df, col_texte)
            if nb > 0: st.success(f"{nb} avis enregistrés avec leur sujet !")
            
            # Petit bouton pour vérifier la BDD
            if st.button("🔍 Vérifier le contenu de la base"):
                session = SessionLocal()
                data = session.query(Review).all()
                session.close()
                st.dataframe([{"Avis": r.text_content, "Topic": r.topic} for r in data])

    else:
        st.error("Colonne texte manquante.")