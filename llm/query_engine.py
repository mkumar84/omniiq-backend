import json
from datetime import datetime, timezone
from .claude_client import call_claude

NL_QUERY_PROMPT = """\
You are OmniIQ's natural language query engine for marketing analytics.
A marketing manager has asked you a question about their business.
You have access to the following pre-computed analytics results.
Answer ONLY from this data. Do not use training knowledge to supplement your answer.
If the answer cannot be found in the data provided, say:
'This question requires data not currently available in OmniIQ.'
Do not guess or infer beyond what the numbers support.

AVAILABLE ANALYTICS DATA:
Segmentation results: {segmentation_json}
Attribution results: {attribution_json}
Churn results: {churn_json}
Campaign results: {campaign_json}

MARKETING MANAGER'S QUESTION: {question}

Answer in 2-3 sentences. Be specific — include numbers from the data.
End with: 'Source: OmniIQ ML Analytics | {timestamp}'\
"""


def answer_query(question: str, cache: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    system = NL_QUERY_PROMPT.format(
        segmentation_json=json.dumps(cache.get("segmentation", []), default=str),
        attribution_json=json.dumps(cache.get("attribution", []), default=str),
        churn_json=json.dumps(
            {
                "top10_atrisk": cache.get("churn_top50", [])[:10],
                "feature_importance": cache.get("churn_feature_importance", {}),
                "churn_rate_pct": round(cache.get("churn_rate", 0), 1),
            },
            default=str,
        ),
        campaign_json=json.dumps(
            {"top_products": cache.get("campaign_top_products", [])}, default=str
        ),
        question=question,
        timestamp=ts,
    )
    return call_claude(system, question, max_tokens=512)
