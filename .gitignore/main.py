import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="QA System")
st.title("🚀 My AI Web App")

# Data
text = ["The course coordinator is Dr. Dibakar Sinha.", "Manipal University Jaipur was established in 2011."]

# Logic
if st.button("Initialize AI"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(text, embeddings)
    st.success("AI is ready!")

query = st.text_input("Ask a question:")
if query:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    st.write("AI is thinking...")
    # This is a simplified direct response for testing
    response = llm.invoke(f"Based on this data: {text}, answer this: {query}")
    st.write(response.content)