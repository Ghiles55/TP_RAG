from sentence_transformers import SentenceTransformer
import chromadb
import json
import torch


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
	# Parcours recursif de l'arbre LEGI pour extraire les articles.
	# - section_path : liste des titres de sections traverses (du racine jusqu'ici)
	# - partie_filter : on ne garde que les articles dont le 1er niveau correspond
	#   (ex: "Partie législative") pour eviter d'avoir les articles abroges/anciens.
	type_node = node.get("type")
	data = node.get("data", {}) or {}

	if type_node == "section":
		titre = (data.get("title", "") or "").strip()
		# Le JSON LEGI met parfois des \r\n en fin de titre, on normalise les espaces
		titre = " ".join(titre.split())
		# On verifie qu'on est bien dans la partie souhaitee (test fait au 1er niveau)
		if len(section_path) == 0 and partie_filter and partie_filter.lower() not in titre.lower():
			return  # on coupe la branche, ce n'est pas la partie qu'on veut
		new_path = section_path + [titre]
		for child in node.get("children", []):
			parse_articles(child, new_path, articles_out, partie_filter)

	elif type_node == "article":
		# On ne garde que les articles en vigueur, avec un texte non vide
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
	# Point d'entree : on appelle parse_articles depuis la racine et on recupere une liste plate.
	articles = []
	for child in legi_root.get("children", []):
		parse_articles(child, [], articles, partie_filter)
	return articles


def build_chunks(articles):
	# Pour chaque article on construit un texte qui contient le numero d'article,
	# le chemin de section et le texte. C'est ce texte qui va etre embedde.
	# Inclure le numero d'article aide la recherche quand l'utilisateur cite un article precis.
	chunks = []
	metadatas = []
	for article in articles:
		# Section courte = derniere section parente (la plus precise), section longue = chemin complet
		section_courte = article["section_path"][-1] if article["section_path"] else ""
		section_complete = " > ".join(article["section_path"])

		texte_chunk = (
			f"Article {article['num']} (section : {section_courte}).\n"
			f"{article['texte']}"
		)
		chunks.append(texte_chunk)
		metadatas.append({
			"article": article["num"],
			"titre": section_courte,           # section parente comme titre lisible
			"section": section_complete,       # chemin complet en metadonnee
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
	# 1. Chargement du modele d'embedding
	# paraphrase-multilingual-mpnet-base-v2 : multilingue, qualite tres superieure
	# a distiluse pour du francais technique/juridique. Dim 768 (vs 512 pour distiluse).
	device = get_best_device()
	print(f"Device utilise pour l'embedding : {device}")
	sentence_transformer_object = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

	# 2. Chargement du corpus LEGI (Code du travail complet)
	# On garde uniquement la "Partie législative" (les fameux articles L...)
	# en etat VIGUEUR -> ~4400 articles, ce qui suffit largement pour repondre
	# aux questions courantes sans gonfler inutilement l'index.
	print("Chargement du JSON LEGI...")
	legi_root = load_legi_json("corpus/LEGITEXT000006072050.json")
	articles = extract_articles(legi_root, partie_filter="Partie législative")
	print(f"Corpus charge : {len(articles)} articles VIGUEUR (Partie législative)")

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
