import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA

# 1. HARDENED CONFIGURATION
MY_KEY = "AIzaSyC8iJeufUfKEmfUWtkJ_qgEzdFLsKxlSgc"
genai.configure(api_key=MY_KEY)

# 2. CUSTOM EMBEDDING WRAPPER (Bypasses the 404 Prefix Bug)
class SimpleGoogleEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key
    def embed_documents(self, texts):
        # Explicit model call to avoid LangChain's internal naming conflicts
        return [genai.embed_content(model="models/text-embedding-004", content=t, task_type="retrieval_document")['embedding'] for t in texts]
    def embed_query(self, text):
        return genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_query")['embedding']

st.set_page_config(page_title="Agentic RAG System", layout="wide")
st.title("🤖 Autonomous PDF Agent")
st.caption("Advanced RAG Implementation | Vaishnavi Karelia")

# 3. PDF PROCESSING & VECTOR STORAGE
uploaded_file = st.file_uploader("Upload Document", type="pdf")

if uploaded_file:
    if "agent" not in st.session_state:
        with st.status("Initializing Autonomous Agent...") as status:
            # Extraction & Chunking
            reader = PdfReader(uploaded_file)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            chunks = splitter.split_text(text)
            
            # Vector Store
            embeddings = SimpleGoogleEmbeddings(MY_KEY)
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            
            # Create the RAG Tool
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=MY_KEY)
            rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
            
            pdf_tool = Tool(
                name="PDF_Search",
                func=rag_chain.run,
                description="Search the uploaded PDF for specific facts and technical details."
            )
            
            # Initialize the Agent with ReAct Logic
            st.session_state.agent = initialize_agent(
                tools=[pdf_tool],
                llm=llm,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True # This enables the 'Thought/Action' trace in terminal!
            )
            status.update(label="✅ Agent Online!", state="complete")

# 4. AGENTIC QA INTERFACE
if "agent" in st.session_state:
    query = st.text_input("Ask the Agent about the document:")
    if query:
        with st.spinner("Agent is reasoning through the context..."):
            # The agent decides when to use the PDF_Search tool
            response = st.session_state.agent.run(query)
            st.write("### Agent's Response:")
            st.info(response)