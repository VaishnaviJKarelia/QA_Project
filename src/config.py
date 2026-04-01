import os
from transformers import BitsAndBytesConfig

# Paths
CHROMA_PATH = os.path.join(os.getcwd(), "chroma")
DATA_PATH = os.path.join(os.getcwd(), "data")  # Adjust to local data path

# Model configuration
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Replace with accessible model
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Prompt template
PROMPT_TEMPLATE = """
En te basant uniquement sur le contexte suivant, 
identifie les éléments pertinents liés à la question, puis rédige une réponse concise,
reformulée avec tes propres mots, facile à comprendre et bien structurée.

Contexte :
{context}

Question :
{question}

**Réponse :**
"""