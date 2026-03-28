import os
import logging
import streamlit as st
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Optional LangChain imports (may not be installed in demo/offline environments)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain.chat_models import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except Exception:
    RecursiveCharacterTextSplitter = None
    FAISS = None
    RetrievalQA = None
    initialize_agent = None
    AgentType = None
    Tool = None
    ChatGoogleGenerativeAI = None
    LANGCHAIN_AVAILABLE = False

# 1. API CONFIGURATION
# Load .env (if present) to simplify local development, then read the API key.
load_dotenv()
MY_KEY = os.environ.get("GOOGLE_API_KEY")
HAS_KEY = bool(MY_KEY)
if HAS_KEY:
    genai.configure(api_key=MY_KEY)


# 2. EMBEDDING WRAPPER (Hardened to avoid 404 prefix bug)
class HardenedGoogleEmbeddings:
    """Minimal Embeddings wrapper exposing embed_documents and embed_query.

    This bypasses the 404 'models/' prefix issue by calling the low-level
    genai.embed_content with the full model path.

    It can optionally log the raw response from genai.embed_content when
    the environment variable LOG_GENAI_EMBED is set to a truthy value.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or MY_KEY
        self.log_raw = os.getenv("LOG_GENAI_EMBED", "false").lower() in ("1", "true", "yes")
        self.logger = logging.getLogger("HardenedGoogleEmbeddings")

    def _maybe_log(self, prefix: str, data):
        if self.log_raw:
            # Use logger.debug so the user can enable debug logging in their environment
            self.logger.debug(f"%s raw embed response: %s", prefix, data)

    def embed_documents(self, texts):
        # genai.embed_content supports lists; returns a structure containing embeddings
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document",
        )
        self._maybe_log("embed_documents", result)

        # Normalize return: a list of embeddings (one per input text)
        # The SDK may return result['embedding'] or result['embeddings']; handle both.
        if isinstance(result, dict) and "embeddings" in result:
            normalized = [r["embedding"] if isinstance(r, dict) and "embedding" in r else r for r in result["embeddings"]]
            return normalized
        if isinstance(result, dict) and "embedding" in result:
            return result["embedding"]
        return result

    def embed_query(self, text):
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query",
        )
        self._maybe_log("embed_query", result)
        if isinstance(result, dict) and "embedding" in result:
            return result["embedding"]
        return result


# Demo fallback: simple deterministic embeddings + in-memory vector store
import hashlib
import math
from typing import List


class DemoEmbeddings:
    """Deterministic, lightweight embedding generator for offline demos.

    Produces a fixed-size vector (128 dims) derived from SHA256 of the text.
    """

    DIM = 128

    def embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = []
        for i in range(self.DIM):
            byte = h[i % len(h)]
            # scale to [-1,1]
            vals.append((byte / 255.0) * 2 - 1)
        # normalize
        norm = math.sqrt(sum(x * x for x in vals)) or 1.0
        return [x / norm for x in vals]


class InMemoryVectorStore:
    """A tiny vector store with cosine-similarity search for demo purposes."""

    def __init__(self, texts: List[str], embeddings: DemoEmbeddings):
        self.texts = texts
        self.emb = embeddings
        self.vectors = [self.emb.embed(t) for t in texts]

    @classmethod
    def from_texts(cls, texts: List[str], embeddings):
        return cls(texts, embeddings)

    def similarity_search(self, query: str, k: int = 3):
        qv = self.emb.embed(query)
        scores = []
        for idx, v in enumerate(self.vectors):
            # cosine similarity
            dot = sum(a * b for a, b in zip(qv, v))
            scores.append((dot, idx))
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:k]:
            # create a simple doc-like object with page_content
            class Doc:
                def __init__(self, text):
                    self.page_content = text

            results.append(Doc(self.texts[idx]))
        return results


# 3. STREAMLIT UI SETUP
st.set_page_config(page_title="Agentic RAG PDF Analyzer", layout="wide")
st.title("📂 Agentic RAG PDF QA (LangChain + Gemini)")
st.caption("Vaishnavi Karelia — Agentic demo using ZERO_SHOT_REACT_DESCRIPTION")

# If the API key is missing, show a friendly Streamlit error page and stop.
if not HAS_KEY:
    st.error(
        "GOOGLE_API_KEY environment variable is required.\nSet it with: export GOOGLE_API_KEY=\"your_key\"\nSee README_DEMO.md for details."
    )
    st.stop()


# Utility: build FAISS store from uploaded PDF
def build_vector_store_from_pdf(uploaded_file) -> FAISS:
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    # Use LangChain splitter when available, otherwise fall back to a simple chunker
    if RecursiveCharacterTextSplitter is not None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
    else:
        # naive chunking by characters
        chunk_size = 1500
        overlap = 200
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = max(0, end - overlap)
    # Choose vector store implementation depending on availability of genai/FAISS
    if GENAI_AVAILABLE:
        embeddings = HardenedGoogleEmbeddings(MY_KEY)
        # FAISS.from_texts will call embeddings.embed_documents internally
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    else:
        demo_emb = DemoEmbeddings()
        vector_store = InMemoryVectorStore.from_texts(chunks, demo_emb)
        return vector_store


# 4. Upload UI and vector store creation
uploaded_file = st.file_uploader("Upload your Project PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Building FAISS vector store and embeddings..."):
        try:
            vstore = build_vector_store_from_pdf(uploaded_file)
            st.session_state.v_store = vstore
            st.success("Vector store built — ready for agent queries.")
        except Exception as e:
            st.error(f"Error while building vector store: {e}")


# 5. Agentic RAG: create PDF_Search tool backed by RetrievalQA and initialize agent
def create_pdf_search_tool(vector_store: FAISS):
    # Create a retriever from FAISS
    # If using FAISS (real mode), adapt as before. If using InMemoryVectorStore (demo),
    # provide a lightweight search implementation that returns the top doc texts.
    if GENAI_AVAILABLE and hasattr(vector_store, "as_retriever"):
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        # Use ChatGoogleGenerativeAI as the LLM for the RetrievalQA chain
        llm_for_qa = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm_for_qa, chain_type="stuff", retriever=retriever)

        def _pdf_search(query: str) -> str:
            # The RetrievalQA chain's run method returns a string answer
            try:
                return qa_chain.run(query)
            except Exception as e:
                return f"PDF_Search error: {e}"
    else:
        # Demo/simple path
        def _pdf_search(query: str) -> str:
            try:
                docs = vector_store.similarity_search(query, k=3)
                context = "\n---\n".join(d.page_content for d in docs)
                return f"DEMO ANSWER (no external LLM).\nTop context segments:\n{context}"
            except Exception as e:
                return f"PDF_Search demo error: {e}"

    tool = Tool(
        name="PDF_Search",
        func=_pdf_search,
        description="Use this tool to answer questions about the uploaded PDF. Input should be a natural language question. Returns a concise answer with sources when available.",
    )

    return tool


# 6. Interaction: Agent initialization and query handling
if "v_store" in st.session_state:
    st.divider()
    user_query = st.text_input("Ask a question about the document content:")

    if user_query:
        with st.spinner("Agent is thinking (you'll see Thought/Action/Observation in terminal)..."):
            try:
                # Build tool from the stored FAISS index
                pdf_tool = create_pdf_search_tool(st.session_state.v_store)

                # Use ChatGoogleGenerativeAI as the agent's core LLM
                agent_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

                tools = [pdf_tool]

                # Initialize a zero-shot react style agent so it can decide to call the PDF tool
                result = None
                if GENAI_AVAILABLE:
                    agent = initialize_agent(
                        tools=tools,
                        llm=agent_llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                    )

                    # Run the agent — this will print Thought/Action/Observation in the terminal when verbose=True
                    result = agent.run(user_query)
                else:
                    # Demo mode: call the PDF_Search tool directly (no external LLM available)
                    result = pdf_tool.func(user_query)

                st.write("### Agent Response")
                st.info(result)

                # Optional: show top matching docs for transparency
                with st.expander("View top document segments used (retriever results)"):
                    docs = st.session_state.v_store.similarity_search(user_query, k=3)
                    for d in docs:
                        st.write(d.page_content)
                        st.divider()

            except Exception as e:
                st.error(f"Agent runtime error: {e}")
else:
    st.info("Please upload a PDF to activate the agentic RAG system.")
