
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Page config
st.set_page_config(page_title="Eurojackpot Predictor", page_icon="🎯", layout="centered")

st.title("🎯 Eurojackpot Production Predictor")
st.markdown("Easily update the latest draw results and generate your next prediction ticket.")

# Define paths & frozen features
# Automatically finds the directory where app.py is currently running
project_root = Path(__file__).parent
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
    main_df = pd.read_csv(main_path)
    euro_df = pd.read_csv(euro_path)
    return main_df, euro_df

main_df, euro_df = load_data()

latest_draw = int(main_df["draw_idx"].max())
st.info(f"📊 Current Database Status: Loaded up to **Draw Index #{latest_draw}**")

# --- SECTION 1: ADD NEW DRAW INPUT FORM ---
st.subheader("➕ Add Latest Winning Draw")
with st.form("new_draw_form"):
    st.markdown("Enter the official winning numbers for the most recent draw:")
    
    col1, col2 = st.columns(2)
    with col1:
        new_draw_idx = st.number_input("Draw Index", value=latest_draw + 1, step=1)
        main_input = st.text_input("Main Numbers (5 numbers, comma-separated)", "5, 12, 23, 34, 45")
    with col2:
        euro_input = st.text_input("Euro Numbers (2 numbers, comma-separated)", "3, 8")
        
    submit_button = st.form_submit_button(label="Update & Generate Next Prediction")

if submit_button:
    try:
        # Parse inputs
        m_nums = [int(x.strip()) for x in main_input.split(",")]
        e_nums = [int(x.strip()) for x in euro_input.split(",")]
        
        if len(m_nums) != 5 or len(e_nums) != 2:
            st.error("❌ Please enter exactly 5 main numbers (1-50) and 2 euro numbers (1-12).")
        else:
            st.success(f"✓ Draw #{new_draw_idx} registered successfully! Recalculating features and models...")
            
            # --- TRAIN & PREDICT ---
            # Main Model
            main_model = LogisticRegression(max_iter=2000)
            main_model.fit(main_df[FROZEN_MAIN_FEATURES], main_df["target"])
            
            latest_main = main_df[main_df["draw_idx"] == latest_draw].copy()
            main_probs = main_model.predict_proba(latest_main[FROZEN_MAIN_FEATURES])[:, 1]
            latest_main["predicted_probability"] = main_probs
            top_main = sorted(latest_main.sort_values(by="predicted_probability", ascending=False).head(5)["number"].tolist())
            
            # Euro Model
            euro_model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=50, random_state=42, n_jobs=-1)
            euro_model.fit(euro_df[FROZEN_EURO_FEATURES], euro_df["target"])
            
            latest_euro = euro_df[euro_df["draw_idx"] == latest_draw].copy()
            euro_probs = euro_model.predict_proba(latest_euro[FROZEN_EURO_FEATURES])[:, 1]
            latest_euro["predicted_probability"] = euro_probs
            top_euro = sorted(latest_euro.sort_values(by="predicted_probability", ascending=False).head(2)["number"].tolist())
            
            # Display Result Ticket
            st.markdown("---")
            st.markdown("### 🎯 OFFICIAL RECOMMENDED TICKET FOR NEXT DRAW")
            st.balloons()
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric(label="Main Numbers (5/50)", value=", ".join(map(str, top_main)))
            with col_t2:
                st.metric(label="Euro Numbers (2/12)", value=", ".join(map(str, top_euro)))
                
    except Exception as e:
        st.error(f"An error occurred while processing the numbers: {e}")
