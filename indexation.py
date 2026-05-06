from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
import json
import torch
import os


load_dotenv()



EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")


def get_db_path(model_name=EMBEDDING_MODEL_NAME):
	safe = model_name.replace("/", "_")  # certains noms HF contiennent un slash
	return f"./code_travail_db_{safe}"


def get_best_device():
	#Bloc de code pour choisir le GPU selon le device, Cuda si GPU Nvidia ou MPS sur mac
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def load_legi_json(json_path):
	with open(json_path, "r", encoding="utf-8") as f:
		return json.load(f)


def parse_articles(node, section_path, articles_out, partie_filter):
	type_node = node.get("type")
	data = node.get("data", {}) or {}

	if type_node == "section":
		titre = (data.get("title", "") or "").strip()
		titre = " ".join(titre.split())
		if len(section_path) == 0 and partie_filter and partie_filter.lower() not in titre.lower():
			return
		new_path = section_path + [titre]
		for child in node.get("children", []):
			parse_articles(child, new_path, articles_out, partie_filter)

	elif type_node == "article":
		if data.get("etat") != "VIGUEUR":
			return
		texte = (data.get("texte") or "").strip()
		num = data.get("num") or ""
		if not texte or not num:
			return
		articles_out.append({
			"num": num,
			"texte": texte,
			"section_path": section_path,
		})


def extract_articles(legi_root, partie_filter="Partie législative"):
	articles = []
	for child in legi_root.get("children", []):
		parse_articles(child, [], articles, partie_filter)
	return articles


def build_chunks(articles):
	chunks = []
	metadatas = []
	for article in articles:
		section_courte = article["section_path"][-1] if article["section_path"] else ""
		section_complete = " > ".join(article["section_path"])

		texte_chunk = (
			f"Article {article['num']} (section : {section_courte}).\n"
			f"{article['texte']}"
		)
		chunks.append(texte_chunk)
		metadatas.append({
			"article": article["num"],
			"titre": section_courte,
			"section": section_complete,
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
	device = get_best_device()
	print(f"Device utilise pour l'embedding : {device}")
	print(f"Modele d'embedding : {EMBEDDING_MODEL_NAME}")
	sentence_transformer_object = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

	print("Chargement du JSON LEGI...")
	legi_root = load_legi_json("corpus/LEGITEXT000006072050.json")
	articles = extract_articles(legi_root, partie_filter="Partie législative")
	print(f"Corpus charge : {len(articles)} articles VIGUEUR (Partie législative)")

	chuncks, metadatas = build_chunks(articles)

	embeddings = get_embeddings(sentence_transformer_object, chuncks)

	db_path = get_db_path()
	print(f"Base vectorielle : {db_path}")
	chroma = chromadb.PersistentClient(path=db_path)
	collection = chroma.get_or_create_collection("code_du_travail")

	collection.upsert(
		ids=[f"article_{meta['article']}" for meta in metadatas],
		documents=chuncks,
		embeddings=embeddings,
		metadatas=metadatas
		)

	print(f"Indexation terminee : {collection.count()} articles dans la base")

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
