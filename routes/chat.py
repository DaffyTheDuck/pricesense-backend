from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import os
from dotenv import load_dotenv
from services.supabase_service import save_chat_message, get_chat_history

load_dotenv()

router = APIRouter()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    analysis: dict
    promotion: dict
    analysis_id: Optional[str] = None  # ← new


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        context = f"""
        You are PriceSense AI — a senior retail pricing strategist.

        A retailer just ran this promotion analysis:

        PROMOTION:
        - Product: {request.promotion.get('product_name')}
        - Industry: {request.promotion.get('industry')}
        - Category: {request.promotion.get('category')}
        - Base Price: ${request.promotion.get('base_price')}
        - Weekly Units: {request.promotion.get('base_units')}
        - Gross Margin: {request.promotion.get('margin')}%
        - Discount: {request.promotion.get('discount')}%
        - Duration: {request.promotion.get('duration_days')} days
        - Promo Type: {request.promotion.get('promo_type')}
        - Competing SKUs: {', '.join(request.promotion.get('competing_skus', []))}

        ANALYSIS RESULTS:
        - Verdict: {request.analysis.get('verdict')}
        - Net Incremental Profit: ${request.analysis.get('net_profit')}
        - Unit Lift: {request.analysis.get('unit_lift')}%
        - Cannibalization Rate: {request.analysis.get('cannibalization_rate')}%
        - Promo ROI: {request.analysis.get('roi')}%
        - Discount Cost: ${request.analysis.get('discount_cost')}
        - AI Recommendation: {request.analysis.get('ai_recommendation')}

        Answer conversationally but precisely. Reference specific numbers.
        If asked "what if" questions, reason through the impact quantitatively.
        Keep answers concise — 2-4 sentences unless more detail is needed.
        Use **bold** for key numbers and conclusions.
        """

        groq_messages = [{"role": "system", "content": context}]

        for msg in request.messages:
            groq_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        response = client.chat.completions.create(
            model=MODEL,
            messages=groq_messages,
            temperature=0.4,
            max_tokens=2048
        )

        reply = response.choices[0].message.content.strip()

        # ── Save to Supabase if we have an analysis_id ──
        if request.analysis_id:
            # Save the last user message
            last_user = next(
                (m for m in reversed(request.messages) if m.role == "user"),
                None
            )
            if last_user:
                save_chat_message(request.analysis_id, "user", last_user.content)

            # Save AI reply
            save_chat_message(request.analysis_id, "assistant", reply)

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/chat/{analysis_id}")
async def get_chat(analysis_id: str):
    """
    Fetch saved chat history for an analysis.
    Called when user revisits a past analysis.
    """
    try:
        messages = get_chat_history(analysis_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch chat: {str(e)}"
        )