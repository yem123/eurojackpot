import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --- ADMIN CONFIGURATION ---
ADMIN_PASSWORD = "221820"  # Change this to your desired password

# Page config
st.set_page_config(page_title="Eurojackpot Predictor", page_icon="🎯", layout="centered")

st.title("🎯 Eurojackpot Production Predictor")
st.markdown("Inspect historical draws, track 12-month frequencies, search combinations, generate predictions, or manage records.")

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
tab_inspect, tab_12m, tab_add = st.tabs([
    "📊 Inspect History & Predict", 
    "📈 Frequency & Combination Search", 
    "➕ Add New Draw"
])

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
                
                with st.spinner("Calculating features and training models..."):
                    if selected_draw_idx in main_df["draw_idx"].values:
                        latest_main = main_df[main_df["draw_idx"] == selected_draw_idx].copy()
                        latest_euro = euro_df[euro_df["draw_idx"] == selected_draw_idx].copy()
                    else:
                        # Dynamic feature generation for newly added draws
                        max_existing_idx = main_df["draw_idx"].max()
                        latest_main = main_df[main_df["draw_idx"] == max_existing_idx].copy()
                        latest_euro = euro_df[euro_df["draw_idx"] == max_existing_idx].copy()
                        
                        latest_main["draw_idx"] = selected_draw_idx
                        latest_euro["draw_idx"] = selected_draw_idx
                        
                        drawn_mains = [int(row_data[f"main_{i}"]) for i in range(1, 6) if f"main_{i}" in row_data]
                        drawn_euros = [int(row_data[f"euro_{i}"]) for i in range(1, 3) if f"euro_{i}" in row_data]
                        
                        # 1. Update Gaps
                        latest_main.loc[latest_main["number"].isin(drawn_mains), "gap_since_last"] = 0
                        latest_main.loc[~latest_main["number"].isin(drawn_mains), "gap_since_last"] += 1
                        
                        latest_euro.loc[latest_euro["number"].isin(drawn_euros), "gap_since_last"] = 0
                        latest_euro.loc[~latest_euro["number"].isin(drawn_euros), "gap_since_last"] += 1

                        # 2. Update Frequencies dynamically for numbers that just hit
                        for col in ["freq_short", "freq_medium", "freq_long", "freq_all"]:
                            if col in latest_main.columns:
                                latest_main.loc[latest_main["number"].isin(drawn_mains), col] += 1
                        for col in ["freq_short", "freq_medium", "freq_long", "freq_all"]:
                            if col in latest_euro.columns:
                                latest_euro.loc[latest_euro["number"].isin(drawn_euros), col] += 1

                    main_model = LogisticRegression(max_iter=2000)
                    main_model.fit(main_df[FROZEN_MAIN_FEATURES], main_df["target"])
                    
                    main_probs = main_model.predict_proba(latest_main[FROZEN_MAIN_FEATURES])[:, 1]
                    latest_main["predicted_probability"] = main_probs
                    top_main = sorted(latest_main.sort_values(by="predicted_probability", ascending=False).head(5)["number"].tolist())
                    
                    euro_model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=50, random_state=42, n_jobs=-1)
                    euro_model.fit(euro_df[FROZEN_EURO_FEATURES], euro_df["target"])
                    
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
                                
                                st.session_state.clean_df.to_csv(clean_draws_path, index=False)
                                st.cache_data.clear()
                                c_df, m_df, e_df = load_data()
                                st.session_state.clean_df = c_df
                                
                                st.success("✓ Draw record updated and saved permanently to disk!")
                                st.rerun()
                            else:
                                st.error("Please provide exactly 5 main and 2 euro numbers.")
                        except Exception as ex:
                            st.error(f"Update failed: {ex}")
                            
                with col_del:
                    st.markdown("#### Delete Draw Record")
                    if st.button("🗑️ Delete This Draw", type="primary"):
                        st.session_state.clean_df = st.session_state.clean_df[st.session_state.clean_df["draw_date"] != selected_date]
                        
                        st.session_state.clean_df.to_csv(clean_draws_path, index=False)
                        st.cache_data.clear()
                        c_df, m_df, e_df = load_data()
                        st.session_state.clean_df = c_df
                        
                        st.success("✓ Draw deleted and changes saved permanently to disk!")
                        st.rerun()
            elif admin_pwd_input:
                st.error("❌ Incorrect password.")

with tab_12m:
    st.subheader("📈 Last 12 Months Frequency & Repeated Winning Draws")
    
    if "draw_date" in clean_df.columns and not clean_df["draw_date"].isna().all():
        max_date = pd.to_datetime(clean_df["draw_date"]).max()
        start_date = max_date - relativedelta(years=1)
        
        filtered_df = clean_df[(pd.to_datetime(clean_df["draw_date"]) >= start_date) & 
                               (pd.to_datetime(clean_df["draw_date"]) <= max_date)].copy()
        
        st.info(f"📅 Analyzing rolling 12-month period: **{start_date.strftime('%B %d, %Y')}** to **{max_date.strftime('%B %d, %Y')}** ({len(filtered_df)} total draws)")
        
        # Calculate Main Frequencies
        main_cols = [c for c in clean_df.columns if c.startswith("main_")]
        if main_cols:
            all_mains = filtered_df[main_cols].values.flatten()
            m_series = pd.Series(all_mains).dropna().astype(int)
            
            st.markdown("### Clean Ranking by Frequency (Main Numbers 1-50)")
            freq_grouped_main = m_series.value_counts()
            frequency_dict_main = {}
            for num, count in freq_grouped_main.items():
                frequency_dict_main.setdefault(count, []).append(num)
                
            for count in sorted(frequency_dict_main.keys(), reverse=True):
                nums_str = ", ".join(map(str, sorted(frequency_dict_main[count])))
                st.markdown(f"**{count} times:** {nums_str}")
        
        st.markdown("---")
        
        # Calculate Euro Frequencies
        euro_cols = [c for c in clean_df.columns if c.startswith("euro_")]
        if euro_cols:
            all_euros = filtered_df[euro_cols].values.flatten()
            e_series = pd.Series(all_euros).dropna().astype(int)
            
            st.markdown("### Clean Ranking — Euro Numbers (1-12)")
            freq_grouped_euro = e_series.value_counts()
            frequency_dict_euro = {}
            for num, count in freq_grouped_euro.items():
                frequency_dict_euro.setdefault(count, []).append(num)
                
            for count in sorted(frequency_dict_euro.keys(), reverse=True):
                nums_str = ", ".join(map(str, sorted(frequency_dict_euro[count])))
                st.markdown(f"**{count} times:** {nums_str}")
                
        st.markdown("---")
        st.markdown("### 🔄 Repeated Winning Draws & Combinations in Last 12 Months")
        
        # 1. Main Numbers Repeated Combinations (5/50)
        st.markdown("#### 🔵 1. Repeated Main Number Combinations (5/50)")
        filtered_df["main_tuple"] = filtered_df[main_cols].apply(lambda row: tuple(sorted([int(x) for x in row if pd.notna(x)])), axis=1)
        main_draw_counts = filtered_df["main_tuple"].value_counts()
        repeated_mains = main_draw_counts[main_draw_counts > 1]
        
        if not repeated_mains.empty:
            st.warning(f"Found {len(repeated_mains)} identical main number combination(s) repeated:")
            for combo, freq in repeated_mains.items():
                matching_rows = filtered_df[filtered_df["main_tuple"] == combo]
                dates_str = ", ".join(matching_rows["draw_date"].astype(str).tolist())
                st.markdown(f"* **Main Numbers {list(combo)}** appeared **{freq} times** on dates: {dates_str}")
        else:
            st.success("✨ No identical main number combinations were repeated in this window.")

        st.markdown("")

        # 2. Euro Numbers Repeated Combinations (2/12)
        st.markdown("#### 🟡 2. Repeated Euro Number Combinations (2/12)")
        filtered_df["euro_tuple"] = filtered_df[euro_cols].apply(lambda row: tuple(sorted([int(x) for x in row if pd.notna(x)])), axis=1)
        euro_draw_counts = filtered_df["euro_tuple"].value_counts()
        repeated_euros = euro_draw_counts[euro_draw_counts > 1]
        
        if not repeated_euros.empty:
            st.warning(f"Found {len(repeated_euros)} identical euro number combination(s) repeated:")
            for combo, freq in repeated_euros.items():
                matching_rows = filtered_df[filtered_df["euro_tuple"] == combo]
                dates_str = ", ".join(matching_rows["draw_date"].astype(str).tolist())
                st.markdown(f"* **Euro Numbers {list(combo)}** appeared **{freq} times** on dates: {dates_str}")
        else:
            st.success("✨ No identical euro number combinations were repeated in this window.")

        # --- 5 SQUARE INPUT FIELDS FOR ENTIRE DATASET SEARCH ---
        st.markdown("---")
        st.markdown("### 🔎 Search Custom Main Combination (Entire Dataset)")
        st.markdown("Enter 5 main numbers below to search if this exact combination has ever occurred in history:")

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            s_n1 = st.number_input("N1", min_value=1, max_value=50, value=1, step=1, key="search_n1")
        with col_s2:
            s_n2 = st.number_input("N2", min_value=1, max_value=50, value=2, step=1, key="search_n2")
        with col_s3:
            s_n3 = st.number_input("N3", min_value=1, max_value=50, value=3, step=1, key="search_n3")
        with col_s4:
            s_n4 = st.number_input("N4", min_value=1, max_value=50, value=4, step=1, key="search_n4")
        with col_s5:
            s_n5 = st.number_input("N5", min_value=1, max_value=50, value=5, step=1, key="search_n5")

        if st.button("Search Entire Dataset"):
            searched_tuple = tuple(sorted([int(s_n1), int(s_n2), int(s_n3), int(s_n4), int(s_n5)]))
            
            if len(set(searched_tuple)) != 5:
                st.warning("⚠️ Please provide 5 distinct main numbers.")
            else:
                full_clean_copy = clean_df.copy()
                full_clean_copy["search_tuple"] = full_clean_copy[main_cols].apply(
                    lambda row: tuple(sorted([int(x) for x in row if pd.notna(x)])), axis=1
                )
                search_matches = full_clean_copy[full_clean_copy["search_tuple"] == searched_tuple]
                
                if not search_matches.empty:
                    st.success(f"🎉 Found matching draw(s) across the entire dataset!")
                    for _, match_row in search_matches.iterrows():
                        d_date = match_row.get("draw_date", "Unknown Date")
                        d_idx = match_row.get("draw_idx", match_row.name)
                        st.markdown(f"- **Draw Date:** `{d_date}` (Index #{d_idx})")
                else:
                    st.info("No numbers found matching this combination in the entire dataset.")

    else:
        st.error("❌ Unable to calculate frequencies due to missing 'draw_date' column.")

with tab_add:
    st.subheader("➕ Register a New Future Draw")
    
    valid_dates = sorted(clean_df["draw_date"].dropna().unique()) if "draw_date" in clean_df.columns else []
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
                    
                    new_row = {"draw_idx": new_idx, "draw_date": new_draw_date}
                    for i, val in enumerate(m_list, start=1):
                        new_row[f"main_{i}"] = val
                    for i, val in enumerate(e_list, start=1):
                        new_row[f"euro_{i}"] = val
                        
                    st.session_state.clean_df = pd.concat([st.session_state.clean_df, pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state.clean_df.to_csv(clean_draws_path, index=False)
                    
                    st.cache_data.clear()
                    c_df, m_df, e_df = load_data()
                    st.session_state.clean_df = c_df
                    
                    st.success(f"✓ Successfully registered and saved future draw for **{new_draw_date.strftime('%B %d, %Y')}**!")
                    st.balloons()
                    st.rerun()
            except Exception as e:
                    st.error(f"Failed to add draw: {e}")