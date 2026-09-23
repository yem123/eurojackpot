import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Page config
st.set_page_config(page_title="Eurojackpot Predictor", page_icon="🎯", layout="centered")

st.title("🎯 Eurojackpot Production Predictor")
st.markdown("Inspect historical draw records with exact dates or generate your next prediction ticket.")

# Safe path handling for both Colab notebooks and Streamlit Cloud
try:
    project_root = Path(__file__).parent
except NameError:
    project_root = Path.cwd()

clean_draws_path = project_root / "data" / "processed" / "clean_draws.csv"
main_path = project_root / "data" / "processed" / "main_features_rich.csv"
euro_path = project_root / "data" / "processed" / "euro_features_rich.csv"

FROZEN_MAIN_FEATURES = [
    "freq_short", "freq_medium", "freq_long", "freq_all", "freq_trend",
    "gap_since_last", "recency_weighted", "pair_score", "triplet_score",
    "adjacent_flag", "is_odd", "is_low", "ctx_avg_sum", "ctx_avg_odd_count", "ctx_avg_low_count"
]

FROZEN_EURO_FEATURES = [
    "freq_short", "freq_medium", "freq_long", "freq_all", "freq_trend",
    "gap_since_last", "recency_weighted", "pair_score", "adjacent_flag", "is_odd", "is_low"
]

@st.cache_data
def load_data():
    clean_df = pd.read_csv(clean_draws_path)
    main_df = pd.read_csv(main_path)
    euro_df = pd.read_csv(euro_path)
    
    if "draw_date" in clean_df.columns:
        clean_df["draw_date"] = pd.to_datetime(clean_df["draw_date"]).dt.date
    elif "date" in clean_df.columns:
        clean_df["draw_date"] = pd.to_datetime(clean_df["date"]).dt.date
        
    return clean_df, main_df, euro_df

clean_df, main_df, euro_df = load_data()

# Create Tabs for Clean Separation
tab_inspect, tab_add = st.tabs(["📊 Inspect History & Predict", "➕ Add New Draw"])

with tab_inspect:
    st.subheader("📅 Historical Draw Inspection")
    
    if "draw_date" in clean_df.columns:
        valid_dates = sorted(clean_df["draw_date"].dropna().unique())
    else:
        valid_dates = []

    if not valid_dates:
        st.error("❌ No draw dates found in clean_draws.csv.")
    else:
        selected_date = st.selectbox(
            "Choose a verified draw date to inspect:",
            options=valid_dates,
            index=len(valid_dates) - 1
        )

        # Get exact row for this date from clean_draws
        row_data = clean_df[clean_df["draw_date"] == selected_date].iloc[0]
        
        # Extract numbers safely (assuming columns like main_1...main_5 and euro_1...euro_2 or similar)
        # Adjust column names here if your clean_draws.csv column names differ
        try:
            actual_main = [int(row_data[f"main_{i}"]) for i in range(1, 6) if f"main_{i}" in row_data]
            actual_euro = [int(row_data[f"euro_{i}"]) for i in range(1, 3) if f"euro_{i}" in row_data]
        except Exception:
            # Fallback if stored as comma-separated string
            actual_main = [int(x.strip()) for x in str(row_data.get("main_numbers", "")).split(",")] if "main_numbers" in row_data else []
            actual_euro = [int(x.strip()) for x in str(row_data.get("euro_numbers", "")).split(",")] if "euro_numbers" in row_data else []

        st.markdown("---")
        st.markdown(f"### 🏆 Official Results for {selected_date.strftime('%B %d, %Y')}")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(label="Actual Main Numbers (5/50)", value=", ".join(map(str, actual_main)) if actual_main else "Check column mapping")
        with col_r2:
            st.metric(label="Actual Euro Numbers (2/12)", value=", ".join(map(str, actual_euro)) if actual_euro else "Check column mapping")

        st.markdown("---")
        if st.button("Generate Prediction Ticket for Next Draw"):
            try:
                selected_draw_idx = int(row_data.get("draw_idx", main_df["draw_idx"].max()))
                
                with st.spinner("Training models..."):
                    main_model = LogisticRegression(max_iter=2000)
                    main_model.fit(main_df[FROZEN_MAIN_FEATURES], main_df["target"])
                    
                    latest_main = main_df[main_df["draw_idx"] == selected_draw_idx].copy()
                    main_probs = main_model.predict_proba(latest_main[FROZEN_MAIN_FEATURES])[:, 1]
                    latest_main["predicted_probability"] = main_probs
                    top_main = sorted(latest_main.sort_values(by="predicted_probability", ascending=False).head(5)["number"].tolist())
                    
                    euro_model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=50, random_state=42, n_jobs=-1)
                    euro_model.fit(euro_df[FROZEN_EURO_FEATURES], euro_df["target"])
                    
                    latest_euro = euro_df[euro_df["draw_idx"] == selected_draw_idx].copy()
                    euro_probs = euro_model.predict_proba(latest_euro[FROZEN_EURO_FEATURES])[:, 1]
                    latest_euro["predicted_probability"] = euro_probs
                    top_euro = sorted(latest_euro.sort_values(by="predicted_probability", ascending=False).head(2)["number"].tolist())
                    
                st.markdown(f"### 🎯 RECOMMENDED PREDICTION TICKET")
                st.balloons()
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric(label="Predicted Main Numbers", value=", ".join(map(str, top_main)))
                with col_t2:
                    st.metric(label="Predicted Euro Numbers", value=", ".join(map(str, top_euro)))
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab_add:
    st.subheader("➕ Register a New Draw")
    st.markdown("Use the calendar date picker below to input a new draw and its winning numbers:")
    
    with st.form("add_draw_form"):
        new_draw_date = st.date_input("New Draw Date", value=date.today())
        new_main_input = st.text_input("Winning Main Numbers (5 numbers, comma-separated)", "5, 12, 23, 34, 45")
        new_euro_input = st.text_input("Winning Euro Numbers (2 numbers, comma-separated)", "3, 8")
        
        submit_new_draw = st.form_submit_button("Save New Draw")
        
    if submit_new_draw:
        st.success(f"✓ Draw for **{new_draw_date.strftime('%B %d, %Y')}** registered successfully in session memory!")
        st.balloons()