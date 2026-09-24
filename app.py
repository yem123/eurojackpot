import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --- ADMIN CONFIGURATION ---
ADMIN_PASSWORD = "221820"  # Change this to your desired password

# Page config
st.set_page_config(page_title="Eurojackpot Predictor", page_icon="🎯", layout="centered")

st.title("🎯 Eurojackpot Production Predictor")
st.markdown("Inspect historical draw records, generate predictions, or securely manage draw databases.")

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

# Initialize session state for persistence during runtime
if "clean_df" not in st.session_state or "main_df" not in st.session_state or "euro_df" not in st.session_state:
    c_df, m_df, e_df = load_data()
    st.session_state.clean_df = c_df
    st.session_state.main_df = m_df
    st.session_state.euro_df = e_df

clean_df = st.session_state.clean_df
main_df = st.session_state.main_df
euro_df = st.session_state.euro_df

# Create Tabs for Navigation
tab_inspect, tab_add = st.tabs(["📊 Inspect History & Predict", "➕ Add New Draw"])

with tab_inspect:
    st.subheader("📅 Historical Draw Inspection & Management")
    
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

        row_data = clean_df[clean_df["draw_date"] == selected_date].iloc[0]
        
        try:
            actual_main = [int(row_data[f"main_{i}"]) for i in range(1, 6) if f"main_{i}" in row_data]
            actual_euro = [int(row_data[f"euro_{i}"]) for i in range(1, 3) if f"euro_{i}" in row_data]
        except Exception:
            actual_main = [int(x.strip()) for x in str(row_data.get("main_numbers", "")).split(",")] if "main_numbers" in row_data else []
            actual_euro = [int(x.strip()) for x in str(row_data.get("euro_numbers", "")).split(",")] if "euro_numbers" in row_data else []

        st.markdown("---")
        st.markdown(f"### 🏆 Official Results for {selected_date.strftime('%B %d, %Y')}")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(label="Actual Main Numbers (5/50)", value=", ".join(map(str, actual_main)) if actual_main else "N/A")
        with col_r2:
            st.metric(label="Actual Euro Numbers (2/12)", value=", ".join(map(str, actual_euro)) if actual_euro else "N/A")

        # --- PREDICTION SECTION ---
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

        # --- ADMIN EDIT / DELETE SECTION ---
        with st.expander("🔐 Admin Controls: Edit or Delete Selected Draw"):
            admin_pwd_input = st.text_input("Enter Admin Password", type="password", key="inspect_admin_pwd")
            
            if admin_pwd_input == ADMIN_PASSWORD:
                st.success("🔓 Admin Access Granted")
                
                col_edit, col_del = st.columns(2)
                
                with col_edit:
                    st.markdown("#### Edit Draw Numbers")
                    edit_main = st.text_input("New Main Numbers (5, comma-separated)", ", ".join(map(str, actual_main)), key="edit_main")
                    edit_euro = st.text_input("New Euro Numbers (2, comma-separated)", ", ".join(map(str, actual_euro)), key="edit_euro")
                    
                    if st.button("Update Draw Record"):
                        try:
                            m_parsed = [int(x.strip()) for x in edit_main.split(",")]
                            e_parsed = [int(x.strip()) for x in edit_euro.split(",")]
                            if len(m_parsed) == 5 and len(e_parsed) == 2:
                                idx_loc = st.session_state.clean_df[st.session_state.clean_df["draw_date"] == selected_date].index
                                for i in range(1, 6):
                                    st.session_state.clean_df.loc[idx_loc, f"main_{i}"] = m_parsed[i-1]
                                for i in range(1, 3):
                                    st.session_state.clean_df.loc[idx_loc, f"euro_{i}"] = e_parsed[i-1]
                                st.success("✓ Draw record updated successfully in session memory!")
                                st.rerun()
                            else:
                                st.error("Please provide exactly 5 main and 2 euro numbers.")
                        except Exception as ex:
                            st.error(f"Update failed: {ex}")
                            
                with col_del:
                    st.markdown("#### Delete Draw Record")
                    if st.button("🗑️ Delete This Draw", type="primary"):
                        st.session_state.clean_df = st.session_state.clean_df[st.session_state.clean_df["draw_date"] != selected_date]
                        st.success("✓ Draw deleted successfully from session memory!")
                        st.rerun()
            elif admin_pwd_input:
                st.error("❌ Incorrect password.")

with tab_add:
    st.subheader("➕ Register a New Future Draw")
    
    # Determine latest reference date
    latest_existing_date = max(valid_dates) if valid_dates else date.today()
    st.info(f"📌 Latest recorded draw in database is on **{latest_existing_date.strftime('%B %d, %Y')}**. New draw dates must be after this date.")
    
    with st.form("add_draw_form"):
        new_draw_date = st.date_input("New Draw Date", value=latest_existing_date)
        new_main_input = st.text_input("Winning Main Numbers (5 numbers, comma-separated)", "5, 12, 23, 34, 45")
        new_euro_input = st.text_input("Winning Euro Numbers (2 numbers, comma-separated)", "3, 8")
        
        st.markdown("---")
        add_admin_pwd = st.text_input("Admin Password Required to Save", type="password")
        
        submit_new_draw = st.form_submit_button("Save New Draw")
        
    if submit_new_draw:
        if add_admin_pwd != ADMIN_PASSWORD:
            st.error("❌ Incorrect admin password. Cannot save new draw.")
        elif new_draw_date <= latest_existing_date:
            st.error(f"❌ Invalid date! New draw date must be **after** the latest recorded date ({latest_existing_date.strftime('%B %d, %Y')}).")
        else:
            try:
                m_list = [int(x.strip()) for x in new_main_input.split(",")]
                e_list = [int(x.strip()) for x in new_euro_input.split(",")]
                
                if len(m_list) != 5 or len(e_list) != 2:
                    st.error("❌ Please provide exactly 5 main numbers and 2 euro numbers.")
                else:
                    new_idx = int(st.session_state.clean_df["draw_idx"].max()) + 1 if "draw_idx" in st.session_state.clean_df.columns else len(st.session_state.clean_df) + 1
                    
                    # Construct new row dict matching clean_draws format
                    new_row = {"draw_idx": new_idx, "draw_date": new_draw_date}
                    for i, val in enumerate(m_list, start=1):
                        new_row[f"main_{i}"] = val
                    for i, val in enumerate(e_list, start=1):
                        new_row[f"euro_{i}"] = val
                        
                    # Append to session state clean_df
                    st.session_state.clean_df = pd.concat([st.session_state.clean_df, pd.DataFrame([new_row])], ignore_index=True)
                    st.success(f"✓ Successfully registered future draw for **{new_draw_date.strftime('%B %d, %Y')}**!")
                    st.balloons()
            except Exception as e:
                    st.error(f"Failed to add draw: {e}")