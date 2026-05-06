from groq import Groq
from dotenv import load_dotenv
import os

import chromadb
from indexation import retrieve, get_best_device

from sentence_transformers import SentenceTransformer


load_dotenv()


def read_file(file_path):
	with open(file_path, "r", encoding="utf-8") as file:
		return file.read()


def build_context(question, sentence_transformer_object, collection):
	# On lit le template du prompt systeme
	context = read_file(file_path="context.txt")

	# On recupere les 3 articles les plus pertinents pour la question
	docs, metas = retrieve(question, sentence_transformer_object, collection, n=3)

	# On formate les chunks de maniere lisible pour le LLM
	chuncks_formates = ""
	for i, (doc, meta) in enumerate(zip(docs[0], metas[0]), start=1):
		chuncks_formates += f"\n[Chunk {i}] (Article {meta['article']} - section {meta['section']})\n{doc}\n"

	# On remplace le placeholder dans le template
	full_context = context.replace("{{Chuncks}}", chuncks_formates)
	return full_context, metas[0]


def answer_question(question, sentence_transformer_object, collection):
	client = Groq(api_key=os.environ["GROQ_API_KEY"])

	system_prompt, sources = build_context(question, sentence_transformer_object, collection)

	chat_completion = client.chat.completions.create(
		messages=[
			{
				"role": "system",
				"content": system_prompt,
			},
			{
				"role": "user",
				"content": question,
			}
		],
		model="llama-3.3-70b-versatile"
	)

	reponse = chat_completion.choices[0].message.content
	return reponse, sources


def main():
	print("Chargement du modele d'embedding et de la base vectorielle...")

	device = get_best_device()
	print(f"Device utilise pour l'embedding : {device}")
	sentence_transformer_object = SentenceTransformer("distiluse-base-multilingual-cased-v2", device=device)
	chroma = chromadb.PersistentClient(path="./code_travail_db")
	collection = chroma.get_or_create_collection("code_du_travail")

	print(f"Base prete : {collection.count()} articles indexes.")
	print("Tapez 'quit' pour quitter.\n")

	while True:
		question = input("Votre question : ").strip()

		if question.lower() in ["quit", "exit", "q"]:
			print("Au revoir !")
			break

		if not question:
			continue

		reponse, sources = answer_question(question, sentence_transformer_object, collection)

		print("\n--- Reponse ---")
		print(reponse)
		print("\n--- Sources utilisees ---")
		for s in sources:
			print(f"- Article {s['article']} : {s['titre']} (section : {s['section']})")
		print()


if __name__ == "__main__":
	main()
