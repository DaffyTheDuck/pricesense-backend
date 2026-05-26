from fastapi import APIRouter, HTTPException, Query
from typing import List
from models.schemas import HistoryItem, SimilarPromotion, PromotionInput
from services.supabase_service import (
    get_history,
    get_analysis_by_id,
    find_similar_by_vector
)
from services.embeddings import get_promotion_embedding

router = APIRouter()


@router.get("/history", response_model=List[HistoryItem])
async def get_analysis_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Fetch paginated list of past analyses.

    GET /api/history?limit=20&offset=0

    Query parameters:
    - limit: how many results to return (1-100, default 20)
    - offset: how many to skip (for pagination)

    Why Query()? FastAPI's Query() lets us validate URL parameters
    the same way Pydantic validates body data.
    ge=1 means "greater than or equal to 1" — built in validation.
    """
    try:
        analyses = get_history(limit=limit, offset=offset)
        return analyses
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history: {str(e)}"
        )


@router.get("/history/{analysis_id}", response_model=dict)
async def get_single_analysis(analysis_id: str):
    """
    Fetch full details of one specific past analysis.

    GET /api/history/some-uuid-here

    The {analysis_id} in the URL is a path parameter —
    FastAPI automatically extracts it and passes it to the function.

    Why return dict instead of AnalysisResult?
    Because stored analyses have extra fields (industry, category,
    base_price etc.) that AnalysisResult doesn't include.
    A plain dict is more flexible here.
    """
    try:
        analysis = get_analysis_by_id(analysis_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found"
            )

        return analysis
    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch analysis: {str(e)}"
        )


@router.post("/similar", response_model=List[SimilarPromotion])
async def get_similar_promotions(data: PromotionInput):
    """
    Given a promotion, find the most similar past promotions
    using vector similarity search.

    POST /api/similar
    Body: PromotionInput (same as analyze endpoint)

    Why POST and not GET?
    Because we're sending a full promotion object in the body.
    GET requests don't have a body — they only have URL parameters.
    When sending complex data, use POST.

    How it works:
    1. Convert the current promotion to an embedding (vector)
    2. Ask pgvector: which stored vectors are closest to this one?
    3. Return those past promotions with their similarity scores

    This is the feature that makes the app feel truly intelligent —
    "We've seen something like this before, here's what happened"
    """
    try:
        # Step 1: embed the incoming promotion
        promotion_dict = data.model_dump()
        embedding = get_promotion_embedding(promotion_dict)

        # Step 2: vector search in Supabase
        similar = find_similar_by_vector(embedding, limit=3)

        # Step 3: return with similarity scores
        return [
            SimilarPromotion(
                id=str(item["id"]),
                product_name=item["product_name"],
                verdict=item["verdict"],
                net_profit=float(item["net_profit"]),
                discount=int(item["discount"]),
                similarity_score=float(item["similarity"])
            )
            for item in similar
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similar promotion search failed: {str(e)}"
        )