from sentence_transformers import SentenceTransformer
import chromadb
import json
import torch


# C'est pour faire en sorte que le téléchargement du modèle d'embedding se fasse dans un repertoire de notre choix
# import os
# os.environ['HF_HOME'] = './models'


def get_best_device():
	# Choix automatique du device pour le modele d'embedding :
	# - cuda si on a un GPU NVIDIA
	# - mps si on est sur un Mac Apple Silicon (M1/M2/M3)
	# - cpu sinon
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def load_corpus(json_path):
	with open(json_path, "r", encoding="utf-8") as f:
		return json.load(f)


def build_chunks(articles):
	# Pour chaque article on construit un texte qui contient le numero d'article,
	# le titre et le texte. C'est ce texte qui va etre embedde.
	# Inclure le numero d'article aide la recherche quand l'utilisateur cite un article precis.
	chunks = []
	metadatas = []
	for article in articles:
		texte_chunk = f"Article {article['article']} - {article['titre']} (section : {article['section']}). {article['texte']}"
		chunks.append(texte_chunk)
		metadatas.append({
			"article": article["article"],
			"titre": article["titre"],
			"section": article["section"]
		})
	return chunks, metadatas


def get_embeddings(sentence_transformer_object, chuncks):
	embeddings = sentence_transformer_object.encode(
		chuncks,
		batch_size=32,
		normalize_embeddings=True,
		show_progress_bar=True
		).tolist()
	return embeddings


def retrieve(question, sentence_transformer_object, collection, n=3):
	embedded_question = get_embeddings(sentence_transformer_object, [question])[0]

	results = collection.query(query_embeddings=[embedded_question], n_results=n)

	return results["documents"], results["metadatas"]



if __name__ == "__main__":
	# 1. Chargement du modele d'embedding (multilingue, fonctionne tres bien en francais)
	device = get_best_device()
	print(f"Device utilise pour l'embedding : {device}")
	sentence_transformer_object = SentenceTransformer("distiluse-base-multilingual-cased-v2", device=device)

	# 2. Chargement du corpus depuis le JSON
	articles = load_corpus("corpus/code_travail.json")
	print(f"Corpus charge : {len(articles)} articles")

	# 3. Construction des chunks et des metadatas
	chuncks, metadatas = build_chunks(articles)

	# 4. Calcul des embeddings (un vecteur par chunk)
	embeddings = get_embeddings(sentence_transformer_object, chuncks)

	# 5. Creation/ouverture de la base vectorielle persistante
	chroma = chromadb.PersistentClient(path="./code_travail_db")
	collection = chroma.get_or_create_collection("code_du_travail")

	# 6. Ajout des chunks dans la collection
	# On utilise upsert plutot que add pour eviter une erreur si on relance le script
	collection.upsert(
		ids=[f"article_{meta['article']}" for meta in metadatas],
		documents=chuncks,
		embeddings=embeddings,
		metadatas=metadatas
		)

	print(f"Indexation terminee : {collection.count()} articles dans la base")

	# 7. Petit test rapide pour verifier que la recherche fonctionne
	print("\n--- Test de recherche ---")
	docs, metas = retrieve(
		"Combien de jours de conges payes ai-je droit par an ?",
		sentence_transformer_object,
		collection,
		n=3
	)
	for doc, meta in zip(docs[0], metas[0]):
		print(f"\n[Article {meta['article']}] {meta['titre']}")
		print(doc[:200] + "...")
