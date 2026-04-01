import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# 2026 modern imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Minimal RAG helper (avoids depending on langchain.chains.RetrievalQA)
def simple_retrieval_qa(llm, retriever, question: str, topk: int = 4) -> str:
    """Retrieve top-k documents and ask the LLM to answer using the context."""
    docs = retriever.get_relevant_documents(question) if hasattr(retriever, 'get_relevant_documents') else retriever(question)
    contexts = []
    for i, d in enumerate(docs[:topk]):
        txt = d.page_content if hasattr(d, 'page_content') else str(d)
        contexts.append(f"[Context {i+1}]\n{txt}")
    prompt = (
        "You are an expert assistant. Use the following extracted document contexts to answer the question.\n\n"
        + "\n\n".join(contexts)
        + f"\n\nQuestion: {question}\nAnswer concisely and cite context indices where appropriate."
    )
    # The ChatGoogleGenerativeAI instance typically exposes a __call__ or generate method; we'll attempt to call it.
    try:
        # Try the high-level call
        resp = llm.generate([{"content": prompt}])
        # Attempt to parse return structure
        if hasattr(resp, 'generations'):
            # Newer style
            return str(resp.generations[0][0].text)
        if isinstance(resp, dict) and 'candidates' in resp:
            return resp['candidates'][0]['content']
        return str(resp)
    except Exception:
        # Fallback to calling the model as a callable
        try:
            out = llm(prompt)
            return str(out)
        except Exception as e:
            return f"LLM invocation failed: {e}"

# Optional: callback handler interfaces vary between langchain versions.
# We implement a minimal local tracer and a wrapper function around agent.run
# that collects the verbose trace into st.session_state['ai_trace'].

def _ensure_trace_list():
    if "ai_trace" not in st.session_state:
        st.session_state.ai_trace = []

def _append_trace(entry_type: str, content: str):
    _ensure_trace_list()
    st.session_state.ai_trace.append({"type": entry_type, "content": content})

class SimpleAgentTracer:
    """A minimal tracer that collects Thought/Action/Observation strings.

    Not a full LangChain callback. We append lines to session_state and
    display them in the UI. This works reliably irrespective of callback API
    variations across LangChain versions.
    """
    def __init__(self):
        _ensure_trace_list()

    def record_thought(self, thought: str):
        _append_trace("Thought", thought)

    def record_action(self, action: str):
        _append_trace("Action", action)

    def record_observation(self, obs: str):
        _append_trace("Observation", obs)

    def clear(self):
        st.session_state.ai_trace = []


import io
import sys

def _capture_agent_run(agent, query: str, tracer: SimpleAgentTracer):
    """Run the agent while capturing stdout prints and map them to tracer calls.

    Many LangChain agents emit verbose traces via stdout. We capture that output,
    then attempt to heuristically parse lines starting with 'Thought', 'Action',
    or 'Observation' and map them into the tracer. We also append the final
    agent response as an Observation if not already present.
    """
    tracer.clear()
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    result = None
    try:
        # Run the agent; this may print verbose steps to stdout
        result = agent.run(query)
    finally:
        # Restore stdout and collect printed trace
        sys.stdout = old_stdout
        printed = buffer.getvalue()
        # Heuristically parse lines
        for line in printed.splitlines():
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("thought"):
                # e.g., 'Thought: ...'
                parts = line.split(":", 1)
                tracer.record_thought(parts[1].strip() if len(parts) > 1 else "")
            elif low.startswith("action"):
                parts = line.split(":", 1)
                tracer.record_action(parts[1].strip() if len(parts) > 1 else "")
            elif low.startswith("observation"):
                parts = line.split(":", 1)
                tracer.record_observation(parts[1].strip() if len(parts) > 1 else "")
            else:
                # If it's not labeled, stash as Observation fallback
                tracer.record_observation(line)

        # Ensure the final LLM result appears in the trace as an Observation
        if result:
            tracer.record_observation(result)
    return result

# 1. API CONFIGURATION
MY_KEY = "AIzaSyC8iJeufUfKEmfUWtkJ_qgEzdFLsKxlSgc"  # Replace with your real key or set via STREAMLIT_SECRETS
genai.configure(api_key=MY_KEY)

# 2. CUSTOM EMBEDDING WRAPPER (Fixes the 404/v1beta errors)
class SimpleGoogleEmbeddings:
    """Small wrapper around Google Vertex GenAI embeddings endpoint.

    Uses models/text-embedding-004 directly to avoid the v1beta namespace bug.
    Provides the minimal methods expected by LangChain vectorstores: embed_documents and embed_query.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Prefer the stable 2026 production model, fall back to older one if needed
        self._preferred_model = "models/text-embedding-005"
        self._fallback_model = "models/text-embedding-004"
        # Determine the best available model once at init to avoid per-call fallbacks
        self._model = self._select_model()

    def _select_model(self) -> str:
        """Select an available embedding model by querying the API if possible.

        Tries the preferred model first, then the fallback. Uses genai.list_models()
        when available to avoid raising errors for unsupported models.
        """
        # Try to list models if the helper exists
        try:
            models_info = None
            if hasattr(genai, "list_models"):
                models_info = genai.list_models()
            elif hasattr(genai, "get_models"):
                models_info = genai.get_models()
            if models_info is not None:
                names = []
                # models_info can be a dict or list depending on the client
                if isinstance(models_info, dict):
                    for m in models_info.get("models", []) or []:
                        if isinstance(m, dict):
                            names.append(m.get("name") or m.get("model"))
                        else:
                            names.append(str(m))
                elif isinstance(models_info, list):
                    for m in models_info:
                        if isinstance(m, dict):
                            names.append(m.get("name") or m.get("model"))
                        else:
                            names.append(str(m))
                else:
                    names = [str(models_info)]

                if self._preferred_model in names:
                    return self._preferred_model
                if self._fallback_model in names:
                    return self._fallback_model
        except Exception:
            # Listing models not supported by this client/version or failed; we'll try embedded probes below
            pass

        # As a last resort, probe by attempting a single embed with each candidate
        for candidate in (self._preferred_model, self._fallback_model):
            try:
                resp = genai.embed_content(model=candidate, content="test", task_type="retrieval_query")
                if isinstance(resp, dict) and "embedding" in resp:
                    return candidate
            except Exception:
                continue

        # If none worked, raise an informative error early
        raise RuntimeError(
            "No compatible embedding model found. Tried models/text-embedding-005 and models/text-embedding-004."
        )
    def _embed_with_fallback(self, content: str, task_type: str):
        """Embed using the selected model determined at init. Preserves task_type."""
        resp = genai.embed_content(model=self._model, content=content, task_type=task_type)
        return resp["embedding"]

    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            embeddings.append(self._embed_with_fallback(t, task_type="retrieval_document"))
        return embeddings

    def embed_query(self, text):
        return self._embed_with_fallback(text, task_type="retrieval_query")

# 3. UI SETUP
st.set_page_config(page_title="Agentic RAG System", layout="wide")
st.title("🤖 Autonomous Document Agent")
st.caption("Final Project | Vaishnavi Karelia | MUJ 2026")

# 4. PDF PROCESSING & AGENT INITIALIZATION
uploaded_file = st.file_uploader("Upload your Project PDF", type="pdf")

if uploaded_file:
    # We use session_state so the agent doesn't "forget" between questions
    if "agent" not in st.session_state:
        with st.status("🏗️ Initializing AI Agent...") as status:
            # Step A: Text Extraction
            reader = PdfReader(uploaded_file)
            # Safely extract text from each page
            pages_text = []
            for p in reader.pages:
                try:
                    t = p.extract_text()
                except Exception:
                    t = None
                if t:
                    pages_text.append(t)
            text = "\n\n".join(pages_text)
            
            # Step B: Semantic Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            chunks = splitter.split_text(text)
            
            # Step C: Vector Indexing (FAISS)
            embeddings = SimpleGoogleEmbeddings(MY_KEY)
            # FAISS expects an embeddings object with embed_documents/embed_query methods
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            
            # Step D: Setup the Agent's Tool
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=MY_KEY)
            # Use a minimal RAG callable to avoid dependency on langchain.chains.RetrievalQA
            retriever = vector_store.as_retriever()
            def rag_chain(query: str) -> str:
                return simple_retrieval_qa(llm, retriever, query)
            
            # pdf_tool as a simple callable wrapper (avoids Tool/initialize_agent API mismatches)
            def pdf_search_callable(query: str) -> str:
                return rag_chain(query)

            # Lightweight proxy to expose a .run method (keeps existing usage intact)
            class SimpleAgentProxy:
                def __init__(self, func):
                    self._func = func

                def run(self, query: str) -> str:
                    return self._func(query)

            # Use the proxy as the 'agent' in session state. This avoids requiring
            # the older initialize_agent API and still lets users query the RAG tool.
            st.session_state.agent = SimpleAgentProxy(pdf_search_callable)
            status.update(label="✅ Agent Ready for Demo!", state="complete")

# 5. CHAT INTERFACE
if "agent" in st.session_state:
    query = st.text_input("Ask the Agent a question about the PDF:")
    if query:
        with st.spinner("Agent is reasoning..."):
            tracer = SimpleAgentTracer()
            response = _capture_agent_run(st.session_state.agent, query, tracer)
            st.write("### AI Response:")
            st.info(response)

            # Display the reasoning trace in an expander
            with st.expander("AI Reasoning Trace", expanded=True):
                cols = st.columns([1, 6, 1])
                with cols[0]:
                    st.write("Type")
                with cols[1]:
                    st.write("Content")
                with cols[2]:
                    if st.button("Clear Trace"):
                        tracer.clear()
                        st.experimental_rerun()

                # Render trace entries with color-coded blocks
                for entry in st.session_state.get("ai_trace", []):
                    ttype = entry.get("type", "").lower()
                    content = entry.get("content", "")
                    if ttype == "thought":
                        st.info(f"**Thought**\n\n{content}")
                    elif ttype == "action":
                        st.success(f"**Action**\n\n{content}")
                    elif ttype == "observation":
                        st.warning(f"**Observation**\n\n{content}")
                    else:
                        # fallback
                        st.write(f"**{entry.get('type','') }**\n\n{content}")
                    st.markdown("---")
else:
    st.info("Upload a document to start the Agentic RAG system.")