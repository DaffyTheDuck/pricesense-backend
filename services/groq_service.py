import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# We use Llama 3.3 70B — Groq's best free model
# 70B means 70 billion parameters — very capable
MODEL = "openai/gpt-oss-120b"


def build_system_prompt() -> str:
    """
    The system prompt sets the persona and rules for the LLM.
    Think of it as the job description you give the AI before
    it starts working.
    
    This runs once per conversation and shapes ALL responses.
    """
    return """
    You are PriceSense AI — a senior retail pricing strategist with 
    deep expertise in promotional analytics, price elasticity, and 
    category management across grocery, specialty food, beverage, 
    snacks, dairy, and bakery sectors.
    
    Your job is to analyze proposed retail promotions and give 
    clear, data-driven recommendations that help mid-market retailers 
    ($50M-$500M revenue) make better pricing decisions.
    
    You always:
    - Ground your analysis in the specific financial metrics provided
    - Consider cannibalization across competing SKUs
    - Think about margin impact, not just volume
    - Give concrete, actionable alternatives when a promotion is risky
    - Sound like a trusted expert advisor, not a generic chatbot
    
    You always respond in valid JSON only. No prose outside the JSON.
    No markdown. No backticks. Pure JSON.
    """


def build_analysis_prompt(promotion: dict, metrics: dict) -> str:
    """
    The user prompt contains the actual data for this specific analysis.
    
    We pass in two things:
    - promotion: what the retailer told us (inputs)
    - metrics: what we already calculated mathematically
    
    The LLM doesn't recalculate numbers — we give it the numbers.
    Its job is to interpret them and generate strategic language.
    
    This separation is important: math in Python, language in LLM.
    Never trust an LLM to do arithmetic reliably.
    """
    competing = ", ".join(promotion.get("competing_skus", [])) or "none"

    return f"""
    A retailer wants to run the following promotion:
    
    PRODUCT DETAILS:
    - Product: {promotion["product_name"]}
    - Industry: {promotion["industry"]}
    - Category: {promotion["category"]}
    - Base price: ${promotion["base_price"]}
    - Weekly baseline units: {promotion["base_units"]}
    - Gross margin: {promotion["margin"]}%
    - Promotion type: {promotion["promo_type"]}
    - Competing SKUs at risk: {competing}
    
    PROMOTION PROPOSED:
    - Discount: {promotion["discount"]}%
    - Duration: {metrics["duration_days"]} days
    - Promotional price: ${metrics["promo_price"]}
    
    PRE-CALCULATED FINANCIAL METRICS:
    - Projected unit lift: +{metrics["unit_lift"]}%
    - Incremental units: +{metrics["incremental_units"]}
    - Cannibalized units: {metrics["cannibalized_units"]}
    - Cannibalization rate: {metrics["cannibalization_rate"]}%
    - Total discount cost: ${metrics["discount_cost"]}
    - Net incremental profit: ${metrics["net_profit"]}
    - Promotion ROI: {metrics["roi"]}%
    
    Based on this analysis, respond with ONLY a JSON object in this exact structure:
    
    {{
        "verdict": "yes" or "no" or "caution",
        "verdict_headline": "one punchy sentence summarizing the verdict",
        "recommendation": "3-4 sentences of strategic advice explaining why, what to watch, and what to do",
        "risks": [
            {{
                "severity": "high" or "medium" or "low",
                "title": "short risk name",
                "description": "one sentence explaining this specific risk for this promotion"
            }}
        ],
        "alternatives": [
            {{
                "discount": integer (a different discount % to consider),
                "rationale": "one sentence on why this alternative is better"
            }}
        ],
        "confidence": "high" or "medium" or "low"
    }}
    
    STRICT RULES — YOU MUST FOLLOW THESE EXACTLY:
    - If net_profit > 0 AND roi > 30: verdict MUST be "yes"
    - If net_profit > 0 AND roi <= 30: verdict MUST be "caution"
    - If net_profit <= 0: verdict MUST be "no"
    - Current net_profit is {metrics["net_profit"]} and roi is {metrics["roi"]}
    - So verdict MUST be: {"yes" if metrics["net_profit"] > 0 and metrics["roi"] > 30 else "caution" if metrics["net_profit"] > 0 else "no"}
    - Do NOT override this verdict under any circumstances
    """


def calculate_base_metrics(promotion: dict) -> dict:

    print(f"DEBUG metrics input: margin={promotion['margin']}, discount={promotion['discount']}, base_price={promotion['base_price']}")

    # Duration — sent directly from frontend
    duration_days = max(1, promotion["duration_days"])
    weeks = duration_days / 7

    # ─── PRICE ELASTICITY ─────────────────────────────────────────
    # Source: Nielsen/IRI retail research averages
    elasticity_map = {
        "nuts": -2.8,
        "dried_fruit": -2.6,
        "seeds": -2.5,
        "trail_mix": -2.7,
        "snacks": -3.2,
        "chips": -3.5,
        "crackers": -2.8,
        "popcorn": -3.0,
        "beverages": -2.2,
        "juice": -2.0,
        "coffee": -2.0,
        "tea": -1.9,
        "dairy": -2.0,
        "cheese": -2.1,
        "yogurt": -2.0,
        "butter": -1.8,
        "bakery": -1.8,
        "bread": -1.7,
        "pastry": -2.0,
        "cookies": -2.5,
        "produce": -2.7,
        "frozen": -2.2,
        "canned": -1.8,
        "condiments": -1.6,
    }
    elasticity = elasticity_map.get(promotion["category"], -2.2)

    # ─── PROMOTIONAL LIFT ──────────────────────────────────────────
    price_drop = promotion["discount"] / 100
    base_lift = -elasticity * price_drop * 100

    # Promo type multipliers
    promo_multipliers = {
        "price_cut": 1.5,
        "tpr": 1.4,
        "bogo": 1.8,
        "bundle": 1.3,
        "loyalty": 0.9,
    }
    multiplier = promo_multipliers.get(promotion["promo_type"], 1.5)
    unit_lift = min(round(base_lift * multiplier, 1), 85)

    # ─── UNITS ────────────────────────────────────────────────────
    base_units_total = round(promotion["base_units"] * weeks)
    promo_units = round(base_units_total * (1 + unit_lift / 100))
    incremental_units = promo_units - base_units_total

    # ─── CANNIBALIZATION ──────────────────────────────────────────
    # Source: IRI category management studies
    competitor_count = len(promotion.get("competing_skus", []))

    if competitor_count == 0:
        cannibalization_rate = 1.5
    elif competitor_count == 1:
        cannibalization_rate = 7.0
    elif competitor_count <= 3:
        cannibalization_rate = 15.0
    else:
        cannibalization_rate = 28.0

    # Promo type adjustments
    if promotion.get("promo_type") == "loyalty":
        cannibalization_rate *= 0.4
    elif promotion.get("promo_type") == "bogo":
        cannibalization_rate *= 1.15

    # Duration adjustment
    if duration_days > 14:
        cannibalization_rate *= 1.2
    elif duration_days <= 3:
        cannibalization_rate *= 0.8

    cannibalization_rate = round(min(cannibalization_rate, 35), 1)
    cannibalized_units = round(incremental_units * (cannibalization_rate / 100))

    # ─── FINANCIALS ───────────────────────────────────────────────
    base_price = promotion["base_price"]
    margin = promotion["margin"] / 100
    promo_price = round(base_price * (1 - price_drop), 2)
    cogs = base_price * (1 - margin)

    base_gp = (base_price - cogs) * base_units_total
    promo_gp = (promo_price - cogs) * promo_units
    incremental_gp = promo_gp - base_gp

    cannibalization_loss = round(cannibalized_units * base_price * margin, 2)
    net_profit = round(incremental_gp - cannibalization_loss, 2)

    discount_cost = round(base_price * price_drop * promo_units, 2)
    roi = round((net_profit / discount_cost * 100), 1) if discount_cost > 0 else 0
    incremental_rev = round((promo_price * promo_units) - (base_price * base_units_total), 2)

    return {
        "duration_days": duration_days,
        "promo_price": promo_price,
        "unit_lift": unit_lift,
        "base_units_total": base_units_total,
        "promo_units": promo_units,
        "incremental_units": incremental_units,
        "cannibalized_units": cannibalized_units,
        "cannibalization_rate": cannibalization_rate,
        "net_profit": net_profit,
        "discount_cost": discount_cost,
        "roi": roi,
        "incremental_rev": incremental_rev,
        "cannibalization_loss": cannibalization_loss,
    }

def get_ai_recommendation(promotion: dict) -> dict:
    # Step 1: Calculate metrics in Python
    metrics = calculate_base_metrics(promotion)

    # Force verdict based on our own math — never trust LLM for this
    if metrics["net_profit"] > 0 and metrics["roi"] > 10:
        forced_verdict = "yes"
    elif metrics["net_profit"] > 0:
        forced_verdict = "caution"
    else:
        forced_verdict = "no"

    # Step 2: Call Groq only for language — not for the verdict
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_analysis_prompt(promotion, metrics)}
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    # Step 3: Parse response
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        ai_output = json.loads(raw)
    except json.JSONDecodeError:
        # If Groq returns malformed JSON, use safe defaults
        ai_output = {
            "verdict_headline": "Analysis complete",
            "recommendation": raw[:500],  # use raw text as fallback
            "risks": [],
            "alternatives": [],
            "confidence": "medium"
        }

    # Step 4: Merge — our verdict overrides LLM verdict always
    return {
        **metrics,
        "verdict": forced_verdict,  # ← our math wins
        "verdict_headline": ai_output.get("verdict_headline", ""),
        "ai_recommendation": ai_output.get("recommendation", ""),
        "risks": ai_output.get("risks", []),
        "alternatives": ai_output.get("alternatives", []),
        "confidence": ai_output.get("confidence", "medium"),
    }