import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np # Pour les dates simulées
from datetime import datetime, timedelta

from src.database import SessionLocal, Review, init_db
from src.preprocessing import clean_text, detect_language
from src.topic_modeling import run_topic_modeling
from src.sentiment import analyze_sentiment

# --- CONFIGURATION ---
st.set_page_config(page_title="Projet Data Science - Othmane", layout="wide")
init_db()

# --- GESTION DES DROITS (AUTH SIMPLE - SEMAINE 9) ---
def check_password():
    """Gère la connexion et retourne le rôle"""
    if 'role' not in st.session_state:
        st.session_state.role = None

    st.sidebar.header("🔐 Connexion")
    user_type = st.sidebar.radio("Rôle", ["Visiteur", "Analyste", "Administrateur"], key="user_radio")
    
    if user_type == "Administrateur":
        pwd = st.sidebar.text_input("Mot de passe Admin", type="password")
        if pwd == "admin123":
            st.sidebar.success("Mode Admin Activé 🔓")
            return "admin"
    elif user_type == "Analyste":
        return "analyste"
    
    return "visiteur"

# --- FONCTIONS UTILITAIRES ---
def sentiment_to_stars(sentiment_label):
    if "Positif" in sentiment_label: return 5
    elif "Neutre" in sentiment_label: return 3
    else: return 1

def save_to_db(df, text_column, source_column='source'):
    session = SessionLocal()
    count = 0
    try:
        for index, row in df.iterrows():
            new_review = Review(
                text_content=row[text_column],
                source=row.get(source_column, "manuel"),
                topic=row.get('Sujet_Dominant', None),
                sentiment=row.get('Sentiment', None),
                score=row.get('Score_IA', 0.0)
            )
            session.add(new_review)
            count += 1
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        st.error(f"Erreur : {e}")
        return 0
    finally:
        session.close()

# =========================================================
# DÉBUT DE L'APPLICATION
# =========================================================

role = check_password()
if role == "visiteur":
    st.warning("🔒 Veuillez sélectionner un rôle (Analyste ou Administrateur) dans la barre latérale pour commencer.")
    st.stop()

st.title(f"🚀 Pipeline d'Analyse (Mode : {role.upper()})")
st.markdown("Ce dashboard visualise étape par étape le traitement de la donnée.")

# --- BARRE LATÉRALE (UPLOAD) ---
st.sidebar.divider()
st.sidebar.header("1. Collecte des Données")
uploaded_file = st.sidebar.file_uploader("Charger un CSV", type=["csv"])

if uploaded_file is not None:
    # Lecture
    df = pd.read_csv(uploaded_file)
    col_trouvee = [c for c in df.columns if c.lower() in ['commentaire', 'avis', 'review', 'text']]
    
    if col_trouvee:
        col_texte = col_trouvee[0]
        st.sidebar.success(f"Colonne détectée : {col_texte}")
        
        # --- TRAITEMENT AUTOMATIQUE (PIPELINE) ---
        with st.spinner('⏳ Exécution du Pipeline IA complet...'):
            
            # 1. Simulation Dates (Pour filtres S9)
            dates_random = [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(len(df))]
            df['Date'] = dates_random
            df['Date'] = pd.to_datetime(df['Date']).dt.date

            # 2. NLP
            df['Langue'] = df[col_texte].apply(detect_language)
            df['Avis_Nettoye'] = df[col_texte].apply(clean_text)
            
            # 3. Sentiment
            sentiments = []
            scores_ia = []
            for text in df[col_texte]:
                label, score, _ = analyze_sentiment(text)
                sentiments.append(label)
                scores_ia.append(score)
            df['Sentiment'] = sentiments
            df['Score_IA'] = scores_ia
            df['Note_Business'] = df['Sentiment'].apply(sentiment_to_stars)
            
            # 4. Topics
            df, topics_display = run_topic_modeling(df, 'Avis_Nettoye', n_topics=2)
            
        st.sidebar.success("✅ Traitement terminé !")

        # =========================================================
        # 🎛️ FILTRES INTERACTIFS (SEMAINE 9)
        # =========================================================
        st.sidebar.divider()
        st.sidebar.header("🎛️ Filtres")
        
        # Filtres
        date_range = st.sidebar.date_input("Période", [df['Date'].min(), df['Date'].max()])
        selected_topics = st.sidebar.multiselect("Thèmes", df['Sujet_Dominant'].unique(), default=df['Sujet_Dominant'].unique())
        selected_sentiments = st.sidebar.multiselect("Sentiments", df['Sentiment'].unique(), default=df['Sentiment'].unique())

        # Application du filtre sur le DataFrame
        mask = (
            (df['Date'] >= date_range[0]) & 
            (df['Date'] <= date_range[1]) & 
            (df['Sujet_Dominant'].isin(selected_topics)) & 
            (df['Sentiment'].isin(selected_sentiments))
        )
        # On crée df_filtered : C'est LUI qu'on va afficher dans les onglets
        df_filtered = df[mask]
        
        st.info(f"🔍 Filtres actifs : {len(df_filtered)} avis affichés sur {len(df)}.")

        # =========================================================
        # AFFICHAGE PAR ONGLETS (STRUCTURE SHOWCASE)
        # =========================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📂 Données (Sem 2)", 
            "🧹 NLP (Sem 3-4)", 
            "🧠 Sémantique (Sem 5)",
            "❤️ Sentiment (Sem 6)",
            "📈 BI & KPIs (Sem 7)"
        ])

        # --- ONGLET 1 : DONNÉES FILTRÉES ---
        with tab1:
            st.header("Semaine 2 : Données & Filtres")
            st.dataframe(df_filtered[[col_texte, 'Date']].head(10), use_container_width=True)

        # --- ONGLET 2 : NETTOYAGE NLP ---
        with tab2:
            st.header("Semaine 3 & 4 : Prétraitement NLP")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("🇺🇸/🇫🇷 **Langue**")
                st.dataframe(df_filtered[['Langue', col_texte]].head(10), use_container_width=True)
            with col_b:
                st.write("🧹 **Résultat**")
                st.dataframe(df_filtered[['Avis_Nettoye']].head(10), use_container_width=True)

        # --- ONGLET 3 : SÉMANTIQUE ---
        with tab3:
            st.header("Semaine 5 : Topic Modeling")
            cols = st.columns(2)
            for i, (sujet, mots) in enumerate(topics_display.items()):
                with cols[i % 2]:
                    st.success(f"**{sujet}**")
                    st.caption(f"Mots-clés : {mots}")
            
            st.dataframe(df_filtered[['Avis_Nettoye', 'Sujet_Dominant']].head(10), use_container_width=True)

        # --- ONGLET 4 : SENTIMENT ---
        with tab4:
            st.header("Semaine 6 : Analyse de Sentiment")
            def color_sentiment(val):
                color = 'green' if 'Positif' in val else 'red' if 'Négatif' in val else 'orange'
                return f'color: {color}'

            st.dataframe(
                df_filtered[['Sentiment', 'Score_IA', col_texte]].style.map(color_sentiment, subset=['Sentiment']),
                use_container_width=True
            )

        # --- ONGLET 5 : BUSINESS INTELLIGENCE (KPIs Dynamiques) ---
        with tab5:
            st.header("Semaine 7 : Dashboard Décisionnel")
            
            if not df_filtered.empty:
                # KPIs recalculés sur df_filtered
                avg = df_filtered['Note_Business'].mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Note Moyenne (Filtrée)", f"{avg:.2f} / 5⭐")
                c2.metric("Avis Sélectionnés", len(df_filtered))
                
                # Graphiques
                c_graph1, c_graph2 = st.columns(2)
                with c_graph1:
                    df_grp = df_filtered.groupby('Sujet_Dominant')['Note_Business'].mean().reset_index()
                    fig = px.bar(df_grp, x='Sujet_Dominant', y='Note_Business', color='Note_Business', title="Qualité par Sujet", range_y=[0,5], color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                
                with c_graph2:
                    fig2 = px.pie(df_filtered, names='Sentiment', title="Répartition", color='Sentiment', color_discrete_map={'Positif 😃':'green', 'Négatif 😡':'red', 'Neutre 😐':'orange'})
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")

        # =========================================================
        # ZONE ADMIN (SEMAINE 9) - EN BAS
        # =========================================================
        # =========================================================
        # ZONE ADMIN (SEMAINE 9 + 10)
        # =========================================================
        st.divider()
        if role == "admin":
            st.header("🛠️ Zone Administrateur")
            
            # --- 1. Gestion Données ---
            st.subheader("💾 Gestion des Données")
            col_admin1, col_admin2 = st.columns(2)
            with col_admin1:
                if st.button("💾 Sauvegarder la sélection en BDD"):
                    nb = save_to_db(df_filtered, col_texte)
                    if nb > 0: st.success(f"{nb} avis archivés.")
            with col_admin2:
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Exporter la sélection (CSV)", csv_data, "export_filtre.csv", "text/csv")
            
            # --- 2. Audit du Modèle (NOUVEAU SEMAINE 10) ---
            st.divider()
            st.subheader("🤖 Audit de Performance IA")
            st.caption("Test du modèle sur un jeu de données de contrôle (Gold Standard).")
            
            if st.button("Lancer l'évaluation du modèle"):
                # On importe la fonction qu'on vient de créer
                from src.evaluation import get_metrics
                
                with st.spinner("Audit en cours..."):
                    acc, f1, nb_test = get_metrics()
                
                # Affichage joli des scores
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Précision (Accuracy)", f"{acc*100:.1f}%")
                kpi2.metric("F1-Score", f"{f1:.2f}")
                kpi3.metric("Echantillon Test", f"{nb_test} phrases")
                
                if acc > 0.7:
                    st.success("✅ Le modèle est performant ( > 70% ).")
                else:
                    st.error("⚠️ Le modèle manque de précision.")

        else:
            st.caption("🔒 Connectez-vous en tant qu'Administrateur pour accéder aux outils techniques.")
    else:
        st.error("Erreur : Colonne texte introuvable.")
else:
    st.info("👈 Veuillez charger un fichier CSV dans le menu à gauche.")