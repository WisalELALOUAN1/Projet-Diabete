# Projet RAG Multi-Thèmes & Classification de Diabète

## 📌 Description
Ce projet combine :
1. **Un système RAG (Retrieval-Augmented Generation)** avec gestion de multiples thèmes
2. **Un module de classification avancée** pour la prédiction de diabète

## 🛠️ Structure du Projet

```
Projet-Diabete/
├── backend/                   # API FastAPI (RAG)
│   ├── api.py                 # Point d'entrée principal
│   └── storage
├── frontend/                  # Interface Streamlit
│   ├── streamlit_app.py       # Application principale
│   
├── classification.ipynb       # Module de classification
│── diabetes.csv
├── .gitignore
├── requirements.txt           # Dépendances globales
└── README.md
```

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/WisalELALOUAN1/Projet-Diabete.git
cd Projet-Diabete

# Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Exécution
**IMPORTANT :** Avant d'exécuter, ajoutez les API keys nécessaires de Hugging Face et Groq.

### 1. Backend (API FastAPI - RAG)

```bash
cd backend
uvicorn api:app --reload
```

➡️ **Accès API** : `http://127.0.0.1:8000/docs`

### 2. Frontend (Streamlit - Interface)

```bash
cd frontend
streamlit run streamlit_app.py
```

➡️ **Accès Interface** : `http://localhost:8501`

### 3. Classification (Jupyter Notebook)
Ouvrez `classification.ipynb` avec Jupyter pour:
* Analyser le dataset `diabetes.csv`
* Exécuter les modèles de classification
* Visualiser les résultats

## 🤖 Modèles Utilisés

### RAG - Modèles LLM (Groq)

| **Modèle** | **Architecture** | **Paramètres** | **Context Window** |
|------------|------------------|----------------|-------------------|
| `DeepSeek-Llama-70B` | Llama-2 + DeepSeek | 70B | 4k tokens |
| `Llama3-8B` | Llama 3 | 8B | 8k tokens |
| `Llama3-70B` | Llama 3 | 70B | 8k tokens |

### RAG - Modèles d'Embedding

| **Modèle** | **Dimensions** | **Usage Recommandé** |
|------------|---------------|----------------------|
| `all-MiniLM-L6-v2` | 384 | Requêtes générales |
| `BAAI/bge-small-en-v1.5` | 384 | Équilibre vitesse/précision |
| `BAAI/bge-base-en-v1.5` | 768 | Recherche complexe |

### Classification - Algorithmes

```python
Algorithms = {
    "Régression Logistique": "Modèle linéaire avec régularisation L1/L2",
    "SVM (NuSVC)": "Version à noyau RBF optimisée",
    "ExtraTrees": "Forêts d'arbres extrêmement randomisés",
    "HistGradientBoosting": "Boost par gradient optimisé",
    "XGBoost": "Version calibrée avec early stopping",
    "LGBM": "Avec paramètres de déséquilibre de classe",
    "CatBoost": "Gestion native des variables catégorielles",
    "MLP": "Réseau de neurones (64-32) optimisé via Optuna"
}
```
