# TP3 - Assistant Code du Travail (RAG)

Petit RAG qui repond a des questions sur le Code du travail francais.

## Stack
- **chromadb** comme base vectorielle (persistante sur disque)
- **sentence-transformers** avec le modele `distiluse-base-multilingual-cased-v2` (multilingue, gere bien le francais)
- **Groq** avec `llama-3.3-70b-versatile` pour la generation
- **python-dotenv** pour la cle API

## Installation

```bash
pip install -r requirements.txt --break-system-packages
```

Cree un fichier `.env` a la racine du dossier avec :
```
GROQ_API_KEY=ta_cle_ici
```

## Utilisation

1. **Indexer le corpus** (a faire une fois, cree le dossier `code_travail_db/`):
```bash
python indexation.py
```

2. **Lancer le RAG en mode interactif** :
```bash
python rag.py
```

Tape ta question puis Entree. Tape `quit` pour sortir.

## Structure du projet

- `corpus/code_travail.json` : ~25 articles couvrant 7 themes (duree du travail, conges payes, contrat, licenciement, rupture conventionnelle, SMIC, harcelement)
- `indexation.py` : charge le corpus, calcule les embeddings, persiste dans chromadb
- `context.txt` : template du prompt systeme avec placeholder `{{Chuncks}}`
- `rag.py` : lit l'index, recherche les 3 articles les plus proches d'une question, demande au LLM de repondre avec les articles comme contexte
- `code_travail_db/` : la base vectorielle persistee (ignoree par git)

## Choix techniques

- **chromadb** plutot que FAISS pour rester dans la lignee du TP1 (api plus simple, persistance integree)
- **distiluse-base-multilingual-cased-v2** : modele d'embedding multilingue, parfait pour du francais juridique
- **Pas de chunking** : un article = un chunk. Les articles du Code du travail sont deja courts et autonomes, decouper introduirait du bruit
- **Numero d'article integre dans le texte embedde** : aide la recherche quand l'utilisateur cite directement un article
- **Prompt systeme strict** : oblige le LLM a citer les articles, refuser de repondre si l'info n'est pas dans le contexte, et a inclure l'avertissement juridique
