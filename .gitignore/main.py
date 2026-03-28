import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the key from the .env file
load_dotenv()
api_key = os.getenv("AIzaSyCSKyxH8AT-YNf0lLRy3QW__jtxW4woaLk")

# 2. Safety Check: Stop if the key is missing before it crashes
if not api_key:
    st.error("❌ API Key not found! Please check your .env file.")
    st.stop()

try:
    # 3. FIX FOR LINE 17: Pass the key DIRECTLY into the embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyCSKyxH8AT-YNf0lLRy3QW__jtxW4woaLk")
    
    # 4. FIX FOR LLM: Do the same for the Chat Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyCSKyxH8AT-YNf0lLRy3QW__jtxW4woaLk")

    st.success("✅ AI Engine Connected!")
    
    # ... (Rest of your UI and logic) ...

except Exception as e:
    st.error(f"Critical Error: {e}")