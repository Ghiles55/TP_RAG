import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

from indexation import get_best_device, EMBEDDING_MODEL_NAME, get_db_path
from rag import answer_question, GROQ_MODEL


# === Configuration de la page ===
st.set_page_config(
	page_title="Assistant Code du Travail",
	page_icon="⚖️",
	layout="wide",
)

st.title("⚖️ Assistant Code du Travail")
st.caption(
	"Pose une question sur le Code du travail francais. "
	"Le RAG va chercher les articles les plus pertinents et te repondre en les citant."
)


# === Chargement des ressources lourdes (mis en cache pour ne pas recharger
# le modele d'embedding et la base a chaque interaction) ===
@st.cache_resource(show_spinner="Chargement du modele d'embedding et de la base vectorielle...")
def load_resources():
	device = get_best_device()
	model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
	chroma = chromadb.PersistentClient(path=get_db_path())
	collection = chroma.get_or_create_collection("code_du_travail")
	return model, collection, device


model, collection, device = load_resources()
nb_articles = collection.count()


# === Barre laterale : etat du systeme ===
with st.sidebar:
	st.header("Etat du systeme")

	if nb_articles == 0:
		st.error(
			"Aucun article indexe.\n\n"
			"Lance d'abord `python indexation.py` dans un terminal pour creer la base."
		)
	else:
		st.success(f"✅ {nb_articles} articles indexes")

	st.write(f"**Device** : `{device}`")
	st.write(f"**Embedding** : `{EMBEDDING_MODEL_NAME}`")
	st.write(f"**LLM Groq** : `{GROQ_MODEL}`")

	st.divider()
	st.caption(
		"⚠️ Cet assistant ne fournit pas de conseil juridique. "
		"Consultez un avocat ou l'inspection du travail pour votre situation personnelle."
	)


# === Zone principale : saisie de la question ===
question = st.text_area(
	"Votre question",
	placeholder="Ex: Combien de temps de preavis pour un licenciement apres 3 ans en CDI ?",
	height=100,
)

ask_disabled = (not question.strip()) or (nb_articles == 0)

if st.button("Demander", type="primary", disabled=ask_disabled):
	with st.spinner("Recherche dans le Code du travail et generation de la reponse..."):
		reponse, sources = answer_question(question, model, collection)

	st.markdown("### Reponse")
	st.markdown(reponse)

	with st.expander(f"📚 Sources utilisees ({len(sources)} articles)", expanded=False):
		for s in sources:
			st.markdown(f"**Article {s['article']}** — {s['titre']}")
			st.caption(s["section"])
			st.divider()
