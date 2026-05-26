from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# What comes IN when user submits a promotion
class PromotionInput(BaseModel):
    product_name: str
    industry: str
    category: str
    base_price: float
    base_units: int
    margin: float
    discount: int
    duration_days: int          # ← direct, clean
    promo_type: str
    competing_skus: List[str] = []

# What goes OUT as the analysis result
class AnalysisResult(BaseModel):
    id: Optional[str] = None
    product_name: str
    verdict: str                    # "yes", "no", "caution"
    net_profit: float
    unit_lift: float
    cannibalization_rate: float
    roi: float
    discount_cost: float
    incremental_units: int
    cannibalized_units: int
    ai_recommendation: str
    risks: List[dict] = []
    alternatives: List[dict] = []
    created_at: Optional[datetime] = None

# What a history item looks like (simpler, for the list view)
class HistoryItem(BaseModel):
    id: str
    product_name: str
    verdict: str
    net_profit: float
    discount: int
    created_at: datetime

# What a similar promotion looks like (from vector search)
class SimilarPromotion(BaseModel):
    id: str
    product_name: str
    verdict: str
    net_profit: float
    discount: int
    similarity_score: float