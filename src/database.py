from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Configuration de la BDD SQLite
# Le fichier sera créé à la racine du projet sous le nom 'avis_clients.db'
DATABASE_URL = "sqlite:///avis_clients.db"

# Création du moteur (le driver)
engine = create_engine(DATABASE_URL, echo=False)

# Création de la Session (pour faire des requêtes)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour nos modèles (classes)
Base = declarative_base()

# 2. Définition de la Table 'Review' (Avis)
class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String)
    text_content = Column(Text)
    topic = Column(String, nullable=True)
    sentiment = Column(String, nullable=True) # Stockera "Positif 😃"
    score = Column(Float, nullable=True)      # <--- CHANGE ICI (Integer -> Float)

# 3. Fonction d'initialisation
def init_db():
    # Crée toutes les tables définies ci-dessus si elles n'existent pas
    Base.metadata.create_all(bind=engine)
    print("Base de données initialisée avec succès !")

# Petit test si on lance ce fichier directement
if __name__ == "__main__":
    init_db()
