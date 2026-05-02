"""
OmniIQ data pipeline — run once offline: python ingest.py
Outputs artifacts/ with trained models and analytics cache.
"""

import itertools
import math
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("data", "online+retail+ii", "online_retail_II.xlsx")
ARTIFACTS_DIR = "artifacts"
CHANNELS = ["Email", "Paid Search", "Organic Search", "Social Media", "Direct"]


# ── 1. Load & clean ──────────────────────────────────────────────────────────

def load_raw():
    print("[1/8] Loading UCI Online Retail II (this may take ~60s) …")
    df1 = pd.read_excel(DATA_PATH, sheet_name="Year 2009-2010", engine="openpyxl")
    df2 = pd.read_excel(DATA_PATH, sheet_name="Year 2010-2011", engine="openpyxl")
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"      Raw rows: {len(df):,}")
    return df


def clean(df_raw):
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names across both UCI sheet variants
    renames = {"Customer ID": "CustomerID", "InvoiceNo": "Invoice", "UnitPrice": "Price"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["IsCancellation"] = df["Invoice"].astype(str).str.startswith("C")

    # Cancellation rate per customer (from full dataset before filtering)
    cancel_stats = (
        df.groupby("CustomerID")
        .agg(total_inv=("Invoice", "nunique"), cancel_inv=("IsCancellation", "sum"))
        .reset_index()
    )
    cancel_stats["CancelRate"] = (
        cancel_stats["cancel_inv"] / cancel_stats["total_inv"].clip(lower=1)
    )

    clean_df = df[~df["IsCancellation"]].copy()
    clean_df = clean_df[(clean_df["Quantity"] > 0) & (clean_df["Price"] > 0)].copy()
    clean_df["Revenue"] = clean_df["Quantity"] * clean_df["Price"]
    clean_df["Hour"] = clean_df["InvoiceDate"].dt.hour
    clean_df["DayOfWeek"] = clean_df["InvoiceDate"].dt.dayofweek  # 0=Mon, 6=Sun

    print(
        f"      Clean rows: {len(clean_df):,}  |  "
        f"Customers: {clean_df['CustomerID'].nunique():,}"
    )
    return clean_df, cancel_stats[["CustomerID", "CancelRate"]]


# ── 2. Channel synthesis ─────────────────────────────────────────────────────

def synthesize_channels(df):
    """Deterministic channel assignment from purchase-time patterns."""
    print("[2/8] Synthesising marketing channels …")

    def _channel(hour, dow):
        if dow >= 5:
            return "Social Media"        # Weekend
        if 6 <= hour < 9:
            return "Email"              # Early morning (email check)
        if 9 <= hour < 12:
            return "Direct"             # Late-morning direct / bookmarked visits
        if 12 <= hour < 15:
            return "Paid Search"        # Midday peak ad traffic
        if 15 <= hour < 18:
            return "Organic Search"     # Afternoon organic browse
        if 18 <= hour < 22:
            return "Social Media"       # Evening social
        return "Direct"                 # Night / edge hours

    df = df.copy()
    df["Channel"] = [_channel(h, d) for h, d in zip(df["Hour"], df["DayOfWeek"])]

    dominant = (
        df.groupby("CustomerID")["Channel"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"Channel": "DominantChannel"})
    )
    channel_revenue = df.groupby("Channel")["Revenue"].sum().reset_index()

    print("      Channel distribution:")
    print("     ", df["Channel"].value_counts().to_dict())
    return df, dominant, channel_revenue


# ── 3. RFM feature engineering ───────────────────────────────────────────────

def build_rfm(df):
    print("[3/8] Building RFM features …")
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    rfm["Recency"] = (snapshot - rfm["LastPurchase"]).dt.days

    # Rank-based scoring avoids qcut duplicate-edge issues
    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] * 0.3 + rfm["F_Score"] * 0.3 + rfm["M_Score"] * 0.4

    return rfm, snapshot


# ── 4. K-Means segmentation ──────────────────────────────────────────────────

def run_segmentation(rfm):
    print("[4/8] Running K-Means segmentation (k=4) …")
    X = rfm[["Recency", "Frequency", "Monetary"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    # Name clusters by descending average RFM score
    order = (
        rfm.groupby("Cluster")["RFM_Score"].mean().sort_values(ascending=False).index.tolist()
    )
    labels = ["Champions", "Loyal", "At Risk", "Lost"]
    name_map = {cid: lbl for cid, lbl in zip(order, labels)}
    rfm["Segment"] = rfm["Cluster"].map(name_map)

    print(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1).to_string())
    return rfm, kmeans, scaler, name_map


# ── 5. Churn prediction ──────────────────────────────────────────────────────

def build_churn_model(df, rfm, dominant, cancel_stats, snapshot):
    print("[5/8] Building churn prediction model …")
    cutoff = snapshot - pd.Timedelta(days=90)

    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
    last_purchase["Churned"] = (last_purchase["InvoiceDate"] < cutoff).astype(int)

    extra = (
        df.groupby("CustomerID")
        .agg(
            TotalRevenue=("Revenue", "sum"),
            TotalOrders=("Invoice", "nunique"),
            UniqueProducts=("StockCode", "nunique"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max"),
        )
        .reset_index()
    )
    extra["AOV"] = extra["TotalRevenue"] / extra["TotalOrders"].clip(lower=1)
    extra["Tenure"] = (extra["LastPurchaseDate"] - extra["FirstPurchase"]).dt.days

    feat = (
        rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]]
        .merge(extra[["CustomerID", "AOV", "UniqueProducts", "Tenure"]], on="CustomerID", how="left")
        .merge(dominant, on="CustomerID", how="left")
        .merge(cancel_stats, on="CustomerID", how="left")
        .merge(last_purchase[["CustomerID", "Churned"]], on="CustomerID", how="left")
    )
    feat["CancelRate"] = feat["CancelRate"].fillna(0)
    feat["Tenure"] = feat["Tenure"].fillna(0)
    feat["DominantChannel"] = feat["DominantChannel"].fillna("Direct")

    le = LabelEncoder()
    feat["ChannelEncoded"] = le.fit_transform(feat["DominantChannel"])

    FEATURES = [
        "Recency", "Frequency", "Monetary", "AOV",
        "CancelRate", "Tenure", "UniqueProducts", "ChannelEncoded",
    ]
    X = feat[FEATURES].fillna(0).values
    y = feat["Churned"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"      AUC-ROC: {auc:.4f}")

    feat["ChurnProba"] = model.predict_proba(X)[:, 1]
    importance = dict(zip(FEATURES, model.feature_importances_))
    return feat, model, le, importance, FEATURES


# ── 6. Shapley value attribution ─────────────────────────────────────────────

def compute_shapley(channel_revenue):
    print("[6/8] Computing Shapley attribution (32 coalitions) …")
    rev_map = dict(zip(channel_revenue["Channel"], channel_revenue["Revenue"]))
    for ch in CHANNELS:
        rev_map.setdefault(ch, 0.0)
    total_rev = sum(rev_map.values())

    # Diminishing-returns value function
    def v(coalition):
        return sum(np.sqrt(rev_map[ch]) for ch in coalition) if coalition else 0.0

    n = len(CHANNELS)
    shapley = {}
    for ch in CHANNELS:
        others = [c for c in CHANNELS if c != ch]
        sv = 0.0
        for size in range(n):
            for sub in itertools.combinations(others, size):
                sub = list(sub)
                w = (
                    math.factorial(size)
                    * math.factorial(n - size - 1)
                    / math.factorial(n)
                )
                sv += w * (v(sub + [ch]) - v(sub))
        shapley[ch] = sv

    total_sv = sum(shapley.values())
    shapley_norm = {ch: sv / total_sv for ch, sv in shapley.items()}

    result = pd.DataFrame({
        "Channel": list(shapley_norm),
        "ShapleyScore": list(shapley_norm.values()),
        "AttributedRevenue": [shapley_norm[ch] * total_rev for ch in shapley_norm],
    }).sort_values("ShapleyScore", ascending=False)

    print(result.to_string(index=False))
    return result


# ── 7. Campaign analytics ─────────────────────────────────────────────────────

def build_campaign_analytics(df):
    print("[7/8] Building campaign analytics …")
    df = df.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    monthly = df.groupby(["YearMonth", "Channel"])["Revenue"].sum().reset_index()
    pivot = (
        monthly.pivot(index="YearMonth", columns="Channel", values="Revenue")
        .fillna(0)
        .reset_index()
    )
    pivot.columns.name = None

    top_products = (
        df.groupby("Description")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    return pivot, top_products


# ── 8. Save artifacts ─────────────────────────────────────────────────────────

def save_artifacts(
    rfm, churn_feat, churn_model, churn_le, churn_importance,
    churn_features, kmeans, scaler, shapley_df,
    monthly_pivot, top_products, channel_revenue,
):
    print("[8/8] Saving artifacts …")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(kmeans, os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(churn_model, os.path.join(ARTIFACTS_DIR, "churn_model.pkl"))
    joblib.dump(churn_le, os.path.join(ARTIFACTS_DIR, "churn_label_encoder.pkl"))

    seg_stats = (
        rfm.groupby("Segment")
        .agg(
            customer_count=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            total_revenue=("Monetary", "sum"),
        )
        .reset_index()
    )
    total_rev_seg = seg_stats["total_revenue"].sum()
    seg_stats["revenue_share_pct"] = (seg_stats["total_revenue"] / total_rev_seg * 100).round(1)

    churn_by_seg = (
        churn_feat.groupby("Segment")
        .agg(
            pct_high_churn=("ChurnProba", lambda x: round(float((x > 0.5).mean() * 100), 1)),
            avg_churn_proba=("ChurnProba", "mean"),
        )
        .reset_index()
    )
    seg_stats = seg_stats.merge(churn_by_seg, on="Segment", how="left")

    top_atrisk = (
        churn_feat.nlargest(50, "ChurnProba")[
            ["CustomerID", "ChurnProba", "Segment", "Monetary", "Recency"]
        ]
        .reset_index(drop=True)
    )

    analytics_cache = {
        "segmentation": seg_stats.to_dict(orient="records"),
        "attribution": shapley_df.to_dict(orient="records"),
        "churn_top50": top_atrisk.to_dict(orient="records"),
        "churn_feature_importance": churn_importance,
        "churn_feature_names": churn_features,
        "campaign_monthly": monthly_pivot.to_dict(orient="records"),
        "campaign_top_products": top_products.to_dict(orient="records"),
        "channel_revenue": channel_revenue.to_dict(orient="records"),
        "total_customers": int(rfm["CustomerID"].nunique()),
        "total_revenue": float(rfm["Monetary"].sum()),
        "avg_order_value": float(churn_feat["AOV"].mean()),
        "churn_rate": float((churn_feat["Churned"] == 1).mean() * 100),
        "top_channel": shapley_df.iloc[0]["Channel"],
        "largest_segment": (
            seg_stats.sort_values("customer_count", ascending=False).iloc[0]["Segment"]
        ),
    }

    joblib.dump(analytics_cache, os.path.join(ARTIFACTS_DIR, "analytics_cache.pkl"))
    print(f"      Saved to {ARTIFACTS_DIR}/")
    return analytics_cache


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  OmniIQ Data Pipeline")
    print("=" * 55)

    raw = load_raw()
    clean_df, cancel_stats = clean(raw)
    channel_df, dominant, channel_revenue = synthesize_channels(clean_df)
    rfm, snapshot = build_rfm(channel_df)
    rfm, kmeans, scaler, name_map = run_segmentation(rfm)
    churn_feat, churn_model, churn_le, churn_importance, churn_features = build_churn_model(
        channel_df, rfm, dominant, cancel_stats, snapshot
    )
    shapley_df = compute_shapley(channel_revenue)
    monthly_pivot, top_products = build_campaign_analytics(channel_df)
    cache = save_artifacts(
        rfm, churn_feat, churn_model, churn_le, churn_importance,
        churn_features, kmeans, scaler, shapley_df,
        monthly_pivot, top_products, channel_revenue,
    )

    print("\n" + "=" * 55)
    print("  Pipeline complete!")
    print(f"  Customers:       {cache['total_customers']:,}")
    print(f"  Total Revenue:   £{cache['total_revenue']:,.2f}")
    print(f"  Avg Order Value: £{cache['avg_order_value']:.2f}")
    print(f"  Churn Rate:      {cache['churn_rate']:.1f}%")
    print(f"  Top Channel:     {cache['top_channel']}")
    print(f"  Largest Segment: {cache['largest_segment']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
