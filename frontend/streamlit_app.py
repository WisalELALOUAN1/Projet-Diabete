import streamlit as st
import requests
import os
from enum import Enum
import logging
from typing import Dict, List

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des clés API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configuration des modèles d'embedding
EMBEDDING_MODELS = {
        "MiniLM (rapide)": "all-MiniLM-L6-v2",
    "BGE Small (équilibré)": "BAAI/bge-small-en-v1.5",
    # Remplacez le modèle suivant par une version qui fonctionne localement
    "BGE Base (qualité)": "BAAI/bge-base-en-v1.5"    # Version simplifiée
}

# Configuration des LLMs
class LLMProvider(Enum):
    GROQ = "Groq"
    

LLM_CONFIG = {
    LLMProvider.GROQ: {
        "models": {
            "DeepSeek-Llama-70B": "deepseek-r1-distill-llama-70b",
            "Llama3-8B": "llama3-8b-8192",
            "Llama3-70B": "llama3-70b-8192"
            
            
        },
        "api_key": "" # Remplacez par votre clé API Groq
    }
}

# Initialisation de l'interface
st.set_page_config(page_title="RAG Multi-Thèmes", page_icon="📚", layout="centered")
st.title("📚 RAG Multi-Thèmes")

# ------------------ Gestion des thèmes ------------------
def load_themes(refresh: bool = False):
    """Charge les thèmes avec option de forcage du rafraîchissement"""
    if refresh or "themes" not in st.session_state:
        try:
            resp = requests.get(f"{API_URL}/themes", timeout=5)
            if resp.status_code == 200:
                themes_data = resp.json().get("themes", [])
                if isinstance(themes_data, dict):  # Cas API retourne un objet
                    st.session_state.themes = list(themes_data.keys())
                else:  # Cas API retourne une liste
                    st.session_state.themes = [t["name"] if isinstance(t, dict) else t for t in themes_data]
            else:
                st.error(f"Erreur API: {resp.text}")
        except Exception as e:
            st.error(f"Erreur de connexion: {str(e)}")
    return st.session_state.get("themes", ["default"])

if "themes" not in st.session_state:
    st.session_state.themes = load_themes()

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Configuration")
    
    # 1. Gestion des thèmes
    current_theme = st.selectbox(
        "Thème actif",
        options=st.session_state.themes,
        help="Sélectionnez un thème pour l'indexation ou le chat"
    )
    
    # Création de nouveau thème
    with st.form("new_theme_form"):
        new_theme = st.text_input("Nom du nouveau thème")
        embedding_model = st.selectbox(
            "Modèle d'embedding", 
            options=list(EMBEDDING_MODELS.keys()),
            index=1
        )
        
        if st.form_submit_button("Créer"):
            if new_theme and new_theme not in st.session_state.themes:
                try:
                    resp = requests.post(
                        f"{API_URL}/theme",
                        json={"name": new_theme, "embedding_model": EMBEDDING_MODELS[embedding_model]}
                    )
                    if resp.status_code == 200:
                        st.session_state.themes = load_themes()
                        st.success(f"Thème '{new_theme}' créé avec succès!")
                    else:
                        st.error(f"Erreur API: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
            else:
                st.error("Le thème existe déjà ou le nom est invalide")

    # 2. Indexation de documents
    st.subheader("⚙️ Indexation")
    uploaded_files = st.file_uploader(
        "Documents à indexer",
        type=["pdf", "txt", "docx", "md", "pptx"],
        accept_multiple_files=True
    )
    
    if st.button("Indexer les documents") and uploaded_files and current_theme:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        try:
            with st.spinner("Indexation en cours..."):
                resp = requests.post(
                    f"{API_URL}/theme/{current_theme}/upload",
                    files=files,
                    timeout=30
                )
                if resp.status_code == 200:
                    st.success(f"{len(uploaded_files)} documents indexés avec succès!")
                else:
                    st.error(f"Erreur API: {resp.json().get('detail', resp.text)}")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            logger.exception("Erreur lors de l'indexation")

    # 3. Configuration du LLM
    st.subheader("🧠 Modèle de langage")
    llm_provider = st.selectbox(
        "Fournisseur",
        options=[p.value for p in LLMProvider],
        index=0
    )
    
    provider_config = LLM_CONFIG[LLMProvider(llm_provider)]
    llm_model = st.selectbox(
        "Modèle",
        options=list(provider_config["models"].keys())
    )

    # Paramètres avancés
    with st.expander("⚙️ Paramètres avancés"):
        temperature = st.slider("Créativité", 0.0, 1.0, 0.7, 0.01)
        max_tokens = st.slider("Longueur max", 100, 4000, 1000, 50)
        system_prompt = st.text_area(
            "Prompt système",
            value="Vous êtes un assistant utile qui répond avec précision et concision.",
            help="Définit le comportement du modèle"
        )
        debug_mode = st.checkbox("Mode débogage")

# ------------------ Zone de chat ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Gestion de la nouvelle question
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Recherche de la réponse..."):
        try:
            payload = {
                "theme": current_theme,
                "question": prompt,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "n_context_results": 3  # Nombre de résultats de contexte
            }
            
            resp = requests.post(
                f"{API_URL}/query",
                json=payload,
                timeout=30
            )
            
            if resp.status_code == 200:
                response_data = resp.json()
                response = response_data.get("answer", "Pas de réponse disponible")
                
                if debug_mode:
                    with st.expander("🔍 Détails de la réponse"):
                        st.json(response_data)
            else:
                error_detail = resp.json().get('detail', resp.text)
                response = f"Erreur API: {error_detail}"
                logger.error(f"Erreur API: {resp.status_code} - {error_detail}")
                
        except Exception as e:
            response = f"Erreur de connexion: {str(e)}"
            logger.exception("Erreur lors de l'appel API")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)

# ------------------ Informations supplémentaires ------------------
with st.sidebar.expander("ℹ️ Informations"):
    st.write(f"**URL API:** {API_URL}")
    st.write(f"**Thème actif:** {current_theme}")
    
    if st.button("Actualiser les thèmes"):
        st.session_state.themes = load_themes()
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            models_info = resp.json()
            st.write("**Modèles disponibles:**")
            st.json(models_info)
    except Exception as e:
        st.warning(f"Impossible de charger les informations: {str(e)}")