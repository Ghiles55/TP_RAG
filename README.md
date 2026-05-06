# TP3 - Assistant Code du Travail (RAG)

Petit RAG qui repond a des questions sur le Code du travail francais.

## Stack
- **chromadb** comme base vectorielle (persistante sur disque)
- **sentence-transformers** avec le modele `paraphrase-multilingual-mpnet-base-v2` (multilingue, dim 768, tres bon en francais juridique)
- **Groq** avec `openai/gpt-oss-120b` pour la generation (meilleur modele de reasoning sur le free tier)
- **python-dotenv** pour la cle API

## Installation

```bash
pip install -r requirements.txt --break-system-packages
```

Cree un fichier `.env` a la racine du dossier avec :
```
# Cle API Groq (obligatoire)
GROQ_API_KEY=ta_cle_ici

# Optionnel : modele LLM Groq (defaut : openai/gpt-oss-120b)
GROQ_MODEL=openai/gpt-oss-120b

# Optionnel : modele d'embedding (defaut : paraphrase-multilingual-mpnet-base-v2)
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

Seul `GROQ_API_KEY` est obligatoire. Si tu omets les deux autres, le code utilise les valeurs par defaut. Tu peux aussi les surcharger ponctuellement en ligne de commande, ex :
```bash
EMBEDDING_MODEL=BAAI/bge-m3 python indexation.py
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

- `corpus/LEGITEXT000006072050.json` : dump LEGI officiel du Code du travail (telecharge depuis [SocialGouv/legi-data](https://github.com/SocialGouv/legi-data)). Arbre hierarchique avec ~11 500 articles.
- `corpus/code_travail.json` : ancien corpus manuel (26 articles), gardé en backup
- `indexation.py` : parse le LEGI (filtre Partie législative + VIGUEUR -> ~4400 articles), calcule les embeddings, persiste dans chromadb
- `context.txt` : template du prompt systeme avec placeholder `{{Chuncks}}`
- `rag.py` : lit l'index, recherche les 10 articles les plus proches d'une question, demande au LLM de repondre avec les articles comme contexte
- `code_travail_db_<modele>/` : la base vectorielle persistee (ignoree par git). Le nom du modele d'embedding est inclus dans le nom du dossier (ex: `code_travail_db_paraphrase-multilingual-mpnet-base-v2/`) pour que la base soit auto-documentee et qu'on puisse avoir plusieurs bases en parallele si on compare des modeles.

## Choix techniques

- **Source des donnees : LEGI officiel** (via SocialGouv/legi-data). On filtre la Partie législative en VIGUEUR -> ~4400 articles (les fameux articles `L...`). La Partie réglementaire (R...) est exclue par défaut pour garder l'index focalisé sur les regles que les utilisateurs citent en pratique.
- **chromadb** plutot que FAISS pour rester dans la lignee du TP1 (api plus simple, persistance integree)
- **paraphrase-multilingual-mpnet-base-v2** : modele d'embedding multilingue (dim 768), nettement meilleur que distiluse sur du francais technique. La 1re version utilisait distiluse (comme le TP1) mais le retrieval etait trop bruite sur du contenu juridique.
- **n_results=10** dans la recherche : les questions juridiques touchent souvent plusieurs sujets (preavis + indemnite + procedure...). Avec n=10, on a une marge confortable pour couvrir tous les angles, et le LLM filtre lui-meme les chunks utiles.
- **Pas de chunking** : un article = un chunk. Les articles du Code du travail ont une mediane de 405 chars (90% < 1200 chars), ce qui rentre largement dans le contexte du modele d'embedding.
- **Numero d'article integre dans le texte embedde** : aide la recherche quand l'utilisateur cite directement un article (ex: "que dit L3141-3 ?")
- **Section parente comme metadonnee** : permet de remonter le chemin hierarchique (Livre > Titre > Chapitre > Section) pour donner du contexte
- **openai/gpt-oss-120b sur Groq** : meilleur modele de reasoning gratuit (mai 2026), suit fidelement les consignes du prompt systeme (citer les articles, refuser quand l'info n'est pas dans la base, ajouter la mention juridique).
- **Detection automatique du device (MPS/CUDA/CPU)** : exploite le GPU Apple Silicon ou NVIDIA si disponible -> indexation 5-10x plus rapide.
- **Prompt systeme strict** : oblige le LLM a citer les articles, refuser de repondre si l'info n'est pas dans le contexte, et a inclure l'avertissement juridique


Question test  qui doit avoir une reponse : 
Je suis en CDI depuis 3 ans et mon employeur veut me licencier pour motif personnel sans faute grave. Quel préavis dois-je effectuer, ai-je droit à une indemnité de licenciement, et quelle est la procédure que mon employeur doit respecter avant de me notifier le licenciement ?

Question test qui ne doit pas avoir de reponse :
Quelle est la procédure exacte pour un divorce par consentement mutuel sans juge, et quels documents dois-je fournir au notaire ?