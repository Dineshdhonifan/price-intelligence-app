import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

st.set_page_config(page_title="Price Intelligence & Recommendations", layout="wide")

@st.cache_data
def generate_sample_data(n_products=8, n_days=60):
    np.random.seed(42)
    random.seed(42)
    products = [
        {"product_id": "P1", "name": "Wireless Earbuds X200", "category": "Electronics", "base_price": 2999},
        {"product_id": "P2", "name": "Smartwatch Pro", "category": "Electronics", "base_price": 5499},
        {"product_id": "P3", "name": "Running Shoes AirFit", "category": "Footwear", "base_price": 3499},
        {"product_id": "P4", "name": "Backpack Urban 30L", "category": "Bags", "base_price": 1899},
        {"product_id": "P5", "name": "Bluetooth Speaker Mini", "category": "Electronics", "base_price": 1499},
        {"product_id": "P6", "name": "Formal Shoes Classic", "category": "Footwear", "base_price": 2799},
        {"product_id": "P7", "name": "Laptop Sleeve 15in", "category": "Bags", "base_price": 899},
        {"product_id": "P8", "name": "Smart LED Bulb Pack", "category": "Electronics", "base_price": 1299},
    ][:n_products]

    rows = []
    start_date = datetime.today() - timedelta(days=n_days)
    for p in products:
        price = p["base_price"]
        for day in range(n_days):
            date = start_date + timedelta(days=day)
            drift = np.random.normal(0, price * 0.005)
            price = max(price + drift, p["base_price"] * 0.5)
            if random.random() < 0.05:
                price *= random.uniform(0.6, 0.85)
            rows.append({
                "product_id": p["product_id"], "name": p["name"],
                "category": p["category"], "date": date.date(), "price": round(price, 2)
            })
    return pd.DataFrame(rows)


def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return generate_sample_data()


def detect_price_drops(df, lookback_days=30, drop_threshold=0.05):
    alerts = []
    for pid, group in df.groupby("product_id"):
        group = group.sort_values("date")
        recent = group.tail(lookback_days)
        avg_price = recent["price"].mean()
        latest_price = group["price"].iloc[-1]
        drop_pct = (avg_price - latest_price) / avg_price
        if drop_pct >= drop_threshold:
            alerts.append({
                "product_id": pid, "name": group["name"].iloc[0],
                "avg_price": round(avg_price, 2), "latest_price": latest_price,
                "drop_pct": round(drop_pct * 100, 1)
            })
    return pd.DataFrame(alerts).sort_values("drop_pct", ascending=False) if alerts else pd.DataFrame()


def buy_recommendation(df, product_id):
    group = df[df["product_id"] == product_id].sort_values("date")
    latest_price = group["price"].iloc[-1]
    historical_min = group["price"].min()
    historical_avg = group["price"].mean()

    if latest_price <= historical_min * 1.02:
        verdict = "🟢 BUY NOW — near all-time low"
    elif latest_price < historical_avg * 0.95:
        verdict = "🟡 GOOD DEAL — below average price"
    else:
        verdict = "🔴 WAIT — price is at/above average"

    return latest_price, historical_min, round(historical_avg, 2), verdict


def recommend_alternatives(df, product_id, n=3):
    latest = df.sort_values("date").groupby("product_id").last().reset_index()
    target = latest[latest["product_id"] == product_id].iloc[0]
    candidates = latest[(latest["category"] == target["category"]) & (latest["product_id"] != product_id)]
    if candidates.empty:
        return pd.DataFrame()
    X = candidates[["price"]].values
    nn = NearestNeighbors(n_neighbors=min(n, len(candidates)))
    nn.fit(X)
    _, idx = nn.kneighbors([[target["price"]]])
    results = candidates.iloc[idx[0]][["product_id", "name", "price"]].copy()
    results["price_diff"] = results["price"] - target["price"]
    return results.sort_values("price")


st.title("🛒 Real-Time Price Intelligence & Recommendation System")
st.caption("Academic project demo — tracks price trends, flags drops, and recommends deals")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload your price_history.csv (optional)", type="csv")
    st.caption("Expected columns: product_id, name, category, date, price")
    st.divider()
    drop_threshold = st.slider("Price drop alert threshold (%)", 1, 30, 5) / 100

df = load_data(uploaded_file)

tab1, tab2, tab3 = st.tabs(["📈 Price Trends", "🔔 Price Drop Alerts", "💡 Recommendations"])

with tab1:
    st.subheader("Price History")
    selected_products = st.multiselect(
        "Select products", df["name"].unique(), default=list(df["name"].unique())[:4]
    )
    filtered = df[df["name"].isin(selected_products)]
    fig = px.line(filtered, x="date", y="price", color="name", markers=True)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Products with Recent Price Drops")
    alerts = detect_price_drops(df, drop_threshold=drop_threshold)
    if alerts.empty:
        st.info("No significant price drops detected at this threshold.")
    else:
        st.dataframe(alerts, use_container_width=True)

with tab3:
    st.subheader("Buy Now / Wait Recommendation")
    product_map = dict(zip(df["name"], df["product_id"]))
    chosen_name = st.selectbox("Choose a product", list(product_map.keys()))
    pid = product_map[chosen_name]

    latest, low, avg, verdict = buy_recommendation(df, pid)
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Price", f"₹{latest}")
    col2.metric("Historical Low", f"₹{low}")
    col3.metric("Historical Avg", f"₹{avg}")
    st.markdown(f"### {verdict}")

    st.divider()
    st.subheader("Similar / Cheaper Alternatives")
    alts = recommend_alternatives(df, pid)
    if alts.empty:
        st.info("No alternatives found in this category.")
    else:
        st.dataframe(alts, use_container_width=True)
