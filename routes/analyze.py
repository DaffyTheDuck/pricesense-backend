from fastapi import APIRouter, HTTPException
from models.schemas import PromotionInput, AnalysisResult
from services.groq_service import get_ai_recommendation
from services.embeddings import get_promotion_embedding
from services.supabase_service import save_analysis, check_cache

# APIRouter is like a mini FastAPI app
# We use routers to keep routes organized by feature
# main.py will collect all routers together
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_promotion(data: PromotionInput):
    """
    Main endpoint — receives a promotion, returns a full analysis.
    
    POST /api/analyze
    Body: PromotionInput (defined in schemas.py)
    Returns: AnalysisResult (defined in schemas.py)
    """

    # Convert Pydantic model to plain dict for easier passing around
    promotion_dict = data.model_dump()

    # ─── STEP 1: CHECK CACHE ───────────────────────────────────────
    # Before doing any expensive work, check if we've already
    # analyzed this exact promotion recently
    cached = check_cache(
        product_name=data.product_name,
        discount=data.discount,
        category=data.category
    )

    if cached:
        # Cache hit — return immediately, no Groq call needed
        # This is fast (milliseconds vs 3-5 seconds)
        return AnalysisResult(
            id=str(cached["id"]),
            product_name=cached["product_name"],
            verdict=cached["verdict"],
            net_profit=cached["net_profit"],
            unit_lift=cached["unit_lift"],
            cannibalization_rate=cached["cannibalization_rate"],
            roi=cached["roi"],
            discount_cost=0,  # not stored, recalculate if needed
            incremental_units=0,
            cannibalized_units=0,
            ai_recommendation=cached["ai_recommendation"] + "\n\n_(Cached result)_",
            created_at=cached["created_at"]
        )

    # ─── STEP 2: GET AI RECOMMENDATION ────────────────────────────
    # This is the expensive step — calls Groq, does all the math
    try:
        result = get_ai_recommendation(promotion_dict)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}"
        )

    # ─── STEP 3: GENERATE EMBEDDING ───────────────────────────────
    # Convert this promotion into a vector for storage and future
    # similarity searches
    try:
        embedding = get_promotion_embedding(promotion_dict)
    except Exception as e:
        # Embedding failure shouldn't kill the whole request
        # We'll save without embedding and skip vector search
        embedding = None

    # ─── STEP 4: SAVE TO DATABASE ─────────────────────────────────
    # Persist everything — the inputs, the outputs, the embedding
    try:
        analysis_data = {
            **promotion_dict,
            "duration_days": result["duration_days"],
            "verdict": result["verdict"],
            "net_profit": result["net_profit"],
            "unit_lift": result["unit_lift"],
            "cannibalization_rate": result["cannibalization_rate"],
            "roi": result["roi"],
            "ai_recommendation": result["ai_recommendation"],
        }
        saved = save_analysis(analysis_data, embedding or [])
        analysis_id = str(saved.get("id", ""))
        print(f"✅ Saved to Supabase with id: {analysis_id}")
    except Exception as e:
        # DB failure shouldn't kill the request either
        # User still gets their result even if saving fails
        print(f"❌ Supabase save failed: {str(e)}")
        analysis_id = ""

    # ─── STEP 5: RETURN RESULT ────────────────────────────────────
    return AnalysisResult(
        id=analysis_id,
        product_name=data.product_name,
        verdict=result["verdict"],
        net_profit=result["net_profit"],
        unit_lift=result["unit_lift"],
        cannibalization_rate=result["cannibalization_rate"],
        roi=result["roi"],
        discount_cost=result["discount_cost"],
        incremental_units=result["incremental_units"],
        cannibalized_units=result["cannibalized_units"],
        ai_recommendation=result["ai_recommendation"],
        risks=result.get("risks", []),
        alternatives=result.get("alternatives", []),
    )