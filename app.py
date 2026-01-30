import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# 3. STORE: Create the Vector Database (The "Brain")
def create_vector_db(chunks):
    # This uses your M4 chip to turn text into numbers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

if __name__ == "__main__":
    print("🚀 Starting RAG Pipeline...")
    
    # 1. Sample Data
    raw_text = "Manipal University Jaipur (MUJ) was established in 2011. It is a premium private university located in Rajasthan. The course coordinator for CS3232 is Dr. Dibakar Sinha."
    
    # 2. Process Data
    chunks = chunk_text(raw_text)
    db = create_vector_db(chunks)
    
    # 3. ASK A QUESTION
    query = "When was Manipal University Jaipur established?"
    
    # Search the vector database for the answer
    docs = db.similarity_search(query)
    
    print("\n--- QUESTION ---")
    print(query)
    print("\n--- RETRIEVED CHUNK ---")
    print(docs[0].page_content)
    print("\n✅ System is ready for LLM integration!")