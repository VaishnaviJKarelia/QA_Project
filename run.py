from src.data_processing import clear_database, load_documents, split_documents, add_to_chroma
from src.model import initialize_model, create_pipeline
from src.interface import create_interface
from src.config import CHROMA_PATH, DATA_PATH

def main():
    print("✨ Clearing Database")
    clear_database(CHROMA_PATH)
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks, CHROMA_PATH)
    
    model, tokenizer = initialize_model()
    pipeline = create_pipeline(model, tokenizer)
    
    interface = create_interface(pipeline)
    interface.launch(share=False)  # Set share=False for local testing

if __name__ == "__main__":
    main()