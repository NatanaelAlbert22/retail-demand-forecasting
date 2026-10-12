import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

@st.cache_resource
def load_model():
    return lgb.Booster(model_file=str(PROJECT_ROOT / "models" / "lgb_forecast_model.txt"))

@st.cache_data
def load_data():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "sales_features.parquet")
    return df

model = load_model()
df = load_data()

st.title("📈 Retail Demand Forecasting Dashboard")
st.markdown("Prediksi permintaan produk berdasarkan data historis penjualan.")

col1, col2 = st.columns(2)
with col1:
    selected_store = st.selectbox("Pilih Toko", sorted(df["store"].unique()))
with col2:
    selected_item = st.selectbox("Pilih Produk", sorted(df["item"].unique()))

filtered = df[(df["store"] == selected_store) & (df["item"] == selected_item)].sort_values("date")

if len(filtered) == 0:
    st.warning("Tidak ada data untuk kombinasi ini.")
else:
    feature_cols = [c for c in df.columns if c not in ["date", "sales"]]
    cat_features = ["store", "item", "day_of_week", "month", "week_of_year"]
    for col in cat_features:
        filtered[col] = filtered[col].astype("category")

    filtered["predicted"] = model.predict(filtered[feature_cols])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["sales"],
        name="Actual", mode="lines", line=dict(color="#4C72B0")
    ))
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["predicted"],
        name="Predicted", mode="lines", line=dict(color="#C44E52", dash="dash")
    ))
    fig.update_layout(
        title=f"Penjualan Aktual vs Prediksi — Toko {selected_store}, Produk {selected_item}",
        xaxis_title="Tanggal", yaxis_title="Unit Terjual", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    recent = filtered.tail(30)
    mae_recent = np.abs(recent["sales"] - recent["predicted"]).mean()
    avg_sales_recent = recent["sales"].mean()

    col1.metric("Rata-rata Penjualan (30 hari terakhir)", f"{avg_sales_recent:.1f} unit")
    col2.metric("MAE (30 hari terakhir)", f"{mae_recent:.2f} unit")
    col3.metric("Prediksi Besok", f"{filtered['predicted'].iloc[-1]:.0f} unit")

st.markdown("---")
st.caption("Model: LightGBM dengan fitur lag dan rolling window, divalidasi dengan time-series cross-validation.")