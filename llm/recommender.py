from .claude_client import call_claude

CAMPAIGN_RECOMMENDER_PROMPT = """\
You are OmniIQ's campaign strategy assistant for retail marketing teams.
You will receive a customer segment profile from a K-Means + RFM model.
Your job is to recommend 3 specific, actionable marketing campaign tactics \
appropriate for this segment's profile.

STRICT RULES:
1. All recommendations must be directly justified by the segment data below.
2. Do not recommend tactics inappropriate for the segment's value tier.
   (e.g. do not suggest high-cost loyalty rewards for the Lost segment)
3. Each recommendation must include: tactic name, rationale, and one specific \
metric to measure success.
4. Do not reference external tools, platforms, or vendors by name.
5. Format as a numbered list of exactly 3 recommendations.

SEGMENT PROFILE:
Segment name: {segment_name}
Customer count: {customer_count}
Avg recency (days): {avg_recency:.1f}
Avg frequency: {avg_frequency:.1f}
Avg monetary (CAD): {avg_monetary:.2f}
Churn risk (% high): {pct_high_churn:.1f}%
Revenue share: {revenue_share_pct:.1f}%\
"""


def recommend_campaigns(seg_data: dict) -> str:
    system = CAMPAIGN_RECOMMENDER_PROMPT.format(
        segment_name=seg_data.get("Segment", "Unknown"),
        customer_count=int(seg_data.get("customer_count", 0)),
        avg_recency=float(seg_data.get("avg_recency", 0)),
        avg_frequency=float(seg_data.get("avg_frequency", 0)),
        avg_monetary=float(seg_data.get("avg_monetary", 0)),
        pct_high_churn=float(seg_data.get("pct_high_churn", 0)),
        revenue_share_pct=float(seg_data.get("revenue_share_pct", 0)),
    )
    return call_claude(system, "Generate 3 campaign recommendations for this segment.", max_tokens=1024)
