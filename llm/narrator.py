import json
from .claude_client import call_claude

SEGMENTATION_NARRATOR_PROMPT = """\
You are OmniIQ's marketing insight narrator. Your role is to translate machine learning \
outputs into plain-English insights for marketing managers who are not data scientists.
You will receive structured segmentation results from a K-Means clustering model.
Your job is to narrate the most important insight in 3-4 sentences.

STRICT RULES:
1. Every claim you make must be directly supported by the numbers provided.
2. Do not add information from your training knowledge about marketing.
3. Do not recommend specific tools, vendors, or platforms.
4. Do not invent statistics not present in the context.
5. Write for a marketing manager, not a data scientist.

SEGMENTATION DATA:
{segmentation_json}

Write your 3-4 sentence insight now. Lead with the most actionable finding.\
"""

ATTRIBUTION_NARRATOR_PROMPT = """\
You are OmniIQ's marketing insight narrator. You will receive Shapley value attribution \
results showing the mathematically fair credit allocation across marketing channels for conversions.

STRICT RULES:
1. Only reference channels and scores present in the data below.
2. Explain what Shapley attribution means in one sentence before the insight.
3. Identify the highest and lowest performing channels specifically.
4. Do not recommend budget changes — surface the insight, not the decision.
5. 3-4 sentences maximum.

ATTRIBUTION DATA:
{attribution_json}\
"""

CHURN_NARRATOR_PROMPT = """\
You are OmniIQ's marketing insight narrator. You will receive churn prediction results \
including the number of at-risk customers, their revenue value, and the features that \
most predict churn.

STRICT RULES:
1. Quantify the revenue at risk using the numbers provided.
2. Name the top 2 churn predictors from feature importance data.
3. Do not recommend specific retention tactics — surface the risk, not the fix.
4. Use urgent but factual language — this is a revenue risk signal.
5. 3-4 sentences maximum.\
"""


def narrate_segmentation(segmentation_data: list[dict]) -> str:
    system = SEGMENTATION_NARRATOR_PROMPT.format(
        segmentation_json=json.dumps(segmentation_data, indent=2, default=str)
    )
    return call_claude(system, "Generate the marketing insight now.")


def narrate_attribution(attribution_data: list[dict]) -> str:
    system = ATTRIBUTION_NARRATOR_PROMPT.format(
        attribution_json=json.dumps(attribution_data, indent=2, default=str)
    )
    return call_claude(system, "Generate the marketing insight now.")


def narrate_churn(churn_data: dict) -> str:
    context = json.dumps(churn_data, indent=2, default=str)
    system = CHURN_NARRATOR_PROMPT + f"\n\nCHURN DATA:\n{context}"
    return call_claude(system, "Generate the marketing insight now.")
