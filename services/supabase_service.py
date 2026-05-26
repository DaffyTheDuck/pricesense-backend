import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional, List

load_dotenv()

# Initialize the Supabase client once, reuse everywhere
# This is called a singleton pattern
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # service role bypasses RLS for backend use
)

def save_analysis(analysis_data: dict, embedding: list) -> dict:
    """
    Save a completed analysis to the database.
    analysis_data contains all the numbers and AI text.
    embedding is the vector representation of this promotion.
    """
    row = {
        "product_name": analysis_data["product_name"],
        "industry": analysis_data["industry"],
        "category": analysis_data["category"],
        "base_price": analysis_data["base_price"],
        "base_units": analysis_data["base_units"],
        "margin": analysis_data["margin"],
        "discount": analysis_data["discount"],
        "promo_type": analysis_data["promo_type"],
        "duration_days": analysis_data["duration_days"],
        "verdict": analysis_data["verdict"],
        "net_profit": analysis_data["net_profit"],
        "unit_lift": analysis_data["unit_lift"],
        "cannibalization_rate": analysis_data["cannibalization_rate"],
        "roi": analysis_data["roi"],
        "ai_recommendation": analysis_data["ai_recommendation"],
        "embedding": embedding  # the vector goes in as a list of floats
    }

    response = supabase.table("analyses").insert(row).execute()
    return response.data[0] if response.data else {}


def get_history(limit: int = 20, offset: int = 0) -> List[dict]:
    """
    Fetch paginated analyses, newest first.
    offset lets us skip rows for pagination.
    """
    response = (
        supabase.table("analyses")
        .select("id, product_name, verdict, net_profit, discount, industry, created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .offset(offset)  # add this line
        .execute()
    )
    return response.data or []


def check_cache(product_name: str, discount: int, category: str) -> Optional[dict]:
    """
    Before running a full analysis, check if we've already done this
    exact promotion recently (within the last 24 hours).
    
    Why 24 hours? Prices and context don't change that fast.
    Returning a cached result is instant vs 3-5 seconds for Groq.
    """
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()

    response = (
        supabase.table("analyses")
        .select("*")
        .eq("product_name", product_name)
        .eq("discount", discount)
        .eq("category", category)
        .gte("created_at", cutoff)  # only within last 24 hours
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0]  # return cached result
    return None  # no cache, proceed with fresh analysis


def find_similar_by_vector(embedding: list, limit: int = 3) -> List[dict]:
    """
    This is the vector search — the most technically impressive part.
    
    We pass in the embedding of the current promotion and ask Supabase
    to find the most similar past promotions using cosine similarity.
    
    Cosine similarity = how close two vectors are in direction.
    Score of 1.0 = identical. Score of 0.0 = completely different.
    
    We use a Postgres RPC (remote procedure call) — a custom SQL
    function we define in Supabase that does the vector math.
    """
    response = supabase.rpc(
        "match_analyses",  # name of our SQL function (we'll create this next)
        {
            "query_embedding": embedding,
            "match_threshold": 0.7,   # only return if similarity > 70%
            "match_count": limit
        }
    ).execute()

    return response.data or []


def get_analysis_by_id(analysis_id: str) -> Optional[dict]:
    """
    Fetch one specific analysis by its UUID.
    Used when the frontend wants to show full details of a past analysis.
    """
    response = (
        supabase.table("analyses")
        .select("*")
        .eq("id", analysis_id)
        .single()  # expects exactly one row
        .execute()
    )
    return response.data

def save_chat_message(analysis_id: str, role: str, content: str) -> dict:
    """
    Save a single chat message linked to an analysis.
    Called after every user message and AI response.
    """
    response = supabase.table("chat_messages").insert({
        "analysis_id": analysis_id,
        "role": role,
        "content": content
    }).execute()
    return response.data[0] if response.data else {}


def get_chat_history(analysis_id: str) -> list:
    """
    Fetch all chat messages for a given analysis.
    Ordered oldest first so conversation flows correctly.
    """
    response = (
        supabase.table("chat_messages")
        .select("role, content, created_at")
        .eq("analysis_id", analysis_id)
        .order("created_at", desc=False)
        .execute()
    )
    return response.data or []