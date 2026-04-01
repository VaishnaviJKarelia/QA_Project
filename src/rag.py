from langchain.vectorstores.chroma import Chroma
from .embeddings import get_embedding_function
from .config import PROMPT_TEMPLATE, CHROMA_PATH

def query_rag(question: str, pipe):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(question, k=5)
    
    context_text = "\n".join([doc.page_content for doc, _ in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    
    response = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, no_repeat_ngram_size=3)[0]["generated_text"]
    
    if "**Réponse :**" in response:
        response_text = response.split("**Réponse :**")[-1].strip()
    else:
        response_text = "Désolé, je n'ai pas pu générer une réponse appropriée."
    
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    sources_text = "Sources :\n" + "\n".join([f"- {src}" for src in sources if src])
    
    return response_text, sources_text