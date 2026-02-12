import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 1. LOAD: Extract text from PDF
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 2. CHUNK: Split text into small pieces
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_text(text)

# 3. STORE: Create the Vector Database
def create_vector_db(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db

if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAQ06eJdI5Rr6P-_zDTIgJRC-xUFzhuucg"
    
    print("🚀 Starting Intelligent QA System...")
    
    # 1. Setup Data
    raw_text = "Manipal University Jaipur (MUJ) was established in 2011. The course coordinator for CS3232 is Dr. Dibakar Sinha."
    chunks = chunk_text(raw_text)
    db = create_vector_db(chunks)
    
    # 2. Setup AI 
    # Using the specific versioned ID (001) often bypasses generic 404 routing errors
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001",
        transport="rest"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    # 3. Ask Question
    query = "Who is the coordinator for CS3232?"
    response = qa_chain.invoke(query)
    
    print("\n--- AI ANSWER ---")
    print(response["result"])
    print("\n✅ System Running Perfectly!")