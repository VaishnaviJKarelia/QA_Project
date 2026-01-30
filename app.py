from pypdf import PdfReader

def load_pdf(file_path):
    # This function reads a PDF and extracts the text
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# For testing, we will just print the first 500 characters
if __name__ == "__main__":
    print("PDF Loader initialized successfully!")