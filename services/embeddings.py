import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Initialize HuggingFace client
# We use the inference API so we don't need to download the model locally
client = InferenceClient(token=os.getenv("HUGGINGFACE_API_KEY"))

# The model we're using for embeddings
# all-MiniLM-L6-v2 is small, fast, free, and excellent for semantic similarity
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_promotion_text(promotion_data: dict) -> str:
    """
    Convert a promotion dict into a rich text description.
    
    Why do we do this instead of just embedding the product name?
    Because meaning comes from context. A 10% discount on a 
    high-margin luxury item is completely different from a 10% 
    discount on a low-margin staple — even if the product names 
    sound similar.
    
    The more context we pack in, the smarter the similarity search.
    """
    competing = ", ".join(promotion_data.get("competing_skus", []))
    
    text = f"""
    Product: {promotion_data.get("product_name")}
    Category: {promotion_data.get("category")}
    Industry: {promotion_data.get("industry")}
    Discount: {promotion_data.get("discount")}%
    Base units per week: {promotion_data.get("base_units")}
    Gross margin: {promotion_data.get("margin")}%
    Promotion type: {promotion_data.get("promo_type")}
    Duration: {promotion_data.get("duration_days")} days
    Competing SKUs: {competing if competing else "none"}
    """.strip()
    
    return text


def get_embedding(text: str) -> List[float]:
    """
    Convert a text string into a vector of 384 floats.
    
    This calls the HuggingFace Inference API — their servers
    run the model and return the vector. We don't run anything
    locally, which keeps our backend lightweight.
    """
    response = client.feature_extraction(
        text=text,
        model=EMBEDDING_MODEL
    )
    
    # HuggingFace returns a nested list, we need a flat list of floats
    # If shape is [[...]] we take the first element
    if isinstance(response, list) and isinstance(response[0], list):
        return response[0]
    
    return list(response)


def get_promotion_embedding(promotion_data: dict) -> List[float]:
    """
    The main function other files will call.
    Takes a promotion dict, builds the text, returns the embedding.
    
    This is the only function supabase_service and analyze.py 
    need to import — they don't need to know the details.
    This is called abstraction.
    """
    text = build_promotion_text(promotion_data)
    embedding = get_embedding(text)
    # Convert float32 to plain Python float so Supabase can serialize it
    return [float(x) for x in embedding]