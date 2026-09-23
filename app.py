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
st.markdown("Inspect historical draw records or add new winning results to update your prediction pipeline.")

# Safe path handling for both Colab notebooks and Streamlit Cloud
try:
    project_root = Path(__file__).parent
except NameError:
    project_root = Path.cwd()

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
    
    if "draw_date" in main_df.columns:
        main_df["draw_date"] = pd.to_datetime(main_df["draw_date"]).dt.date
    if "draw_date" in euro_df.columns:
        euro_df["draw_date"] = pd.to_datetime(euro_df["draw_date"]).dt.date
        
    return main_df, euro_df

# Initialize session state for data so user additions persist during the session
if "main_df" not in st.session_state or "euro_df" not in st.session_state:
    m_df, e_df = load_data()
    st.session_state.main_df = m_df
    st.session_state.euro_df = e_df

main_df = st.session_state.main_df
euro_df = st.session_state.euro_df

# Create Tabs for Clean Separation
tab_inspect, tab_add = st.tabs(["📊 Inspect History & Predict", "➕ Add New Draw"])

with tab_inspect:
    st.subheader("📅 Historical Draw Inspection")
    
    if "draw_date" in main_df.columns:
        valid_dates = sorted(main_df["draw_date"].dropna().unique())
    else:
        valid_dates = []

    if not valid_dates:
        st.error("❌ No draw dates found in the dataset.")
    else:
        selected_date = st.selectbox(
            "Choose a verified draw date to inspect:",
            options=valid_dates,
            index=len(valid_dates) - 1
        )

        # Extract actual winning numbers
        actual_main = sorted(main_df[(main_df["draw_date"] == selected_date) & (main_df["target"] == 1)]["number"].tolist())
        actual_euro = sorted(euro_df[(euro_df["draw_date"] == selected_date) & (euro_df["target"] == 1)]["number"].tolist())

        st.markdown("---")
        st.markdown(f"### 🏆 Official Results for {selected_date.strftime('%B %d, %Y')}")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(label="Actual Main Numbers (5/50)", value=", ".join(map(str, actual_main)) if actual_main else "N/A")
        with col_r2:
            st.metric(label="Actual Euro Numbers (2/12)", value=", ".join(map(str, actual_euro)) if actual_euro else "N/A")

        st.markdown("---")
        if st.button("Generate Prediction Ticket for Next Draw"):
            try:
                selected_draw_idx = int(main_df[main_df["draw_date"] == selected_date]["draw_idx"].max())
                
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
        try:
            m_list = [int(x.strip()) for x in new_main_input.split(",")]
            e_list = [int(x.strip()) for x in new_euro_input.split(",")]
            
            if len(m_list) != 5 or len(e_list) != 2:
                st.error("❌ Please provide exactly 5 main numbers and 2 euro numbers.")
            else:
                # Determine new draw index
                new_idx = int(main_df["draw_idx"].max()) + 1
                
                # Create dummy rows or append targets for the new draw
                # (For demo/session update, we append target=1 rows for the winning numbers)
                new_main_rows = []
                for num in range(1, 51):
                    row = main_df.iloc[0].copy()
                    row["draw_idx"] = new_idx
                    row["draw_date"] = new_draw_date
                    row["number"] = num
                    row["target"] = 1 if num in m_list else 0
                    new_main_rows.append(row)
                    
                new_euro_rows = []
                for num in range(1, 13):
                    row = euro_df.iloc[0].copy()
                    row["draw_idx"] = new_idx
                    row["draw_date"] = new_draw_date
                    row["number"] = num
                    row["target"] = 1 if num in e_list else 0
                    new_euro_rows.append(row)
                    
                # Append to session state dataframes
                st.session_state.main_df = pd.concat([main_df, pd.DataFrame(new_main_rows)], ignore_index=True)
                st.session_state.euro_df = pd.concat([euro_df, pd.DataFrame(new_euro_rows)], ignore_index=True)
                
                st.success(f"✓ Successfully registered draw for **{new_draw_date.strftime('%B %d, %Y')}** (Draw Index #{new_idx})!")
                st.balloons()
        except Exception as e:
            st.error(f"Failed to add draw: {e}")