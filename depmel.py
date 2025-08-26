import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models & scaler
iso_forest = joblib.load("iso_forest_model.pkl")
lof = joblib.load("lof_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Census Anomaly Detection", layout="wide")
st.title("📊 Census Anomaly Detection Dashboard")

# Upload data
uploaded_file = st.file_uploader("Upload CSV with municipality metrics", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Scale numeric features
    features = df.iloc[:, 3:]  # assuming first 3 cols = province/municipality names
    X_scaled = scaler.transform(features.values)

    # Predictions
    iso_predictions = iso_forest.predict(X_scaled)
    lof_predictions = lof.fit_predict(X_scaled)

    # Z-scores
    z_scores = np.abs((features - features.mean()) / features.std(ddof=0))
    z_score_anomalies = (z_scores > 2.0).any(axis=1)  # threshold at 2

    # Add results to df
    df['Isolation_Forest_Score'] = iso_forest.decision_function(X_scaled)
    df['Isolation_Forest_Anomaly'] = iso_predictions
    df['LOF_Anomaly'] = lof_predictions
    df['Z_Score_Anomaly'] = z_score_anomalies

    # Overall anomaly score
    df['Overall_Anomaly_Score'] = (
        (df['Isolation_Forest_Score'] - df['Isolation_Forest_Score'].min()) /
        (df['Isolation_Forest_Score'].max() - df['Isolation_Forest_Score'].min()) +
        (lof.negative_outlier_factor_ - lof.negative_outlier_factor_.min()) /
        (lof.negative_outlier_factor_.max() - lof.negative_outlier_factor_.min()) +
        z_score_anomalies.astype(int)
    ) / 3

    # Sidebar menu
    st.sidebar.header("🔎 Explore Anomalies")
    view_option = st.sidebar.radio("Select View:", ["Top Anomalies", "Search Municipality"])

    # ---- TOP ANOMALIES VIEW ----
    if view_option == "Top Anomalies":
        top_n = st.sidebar.slider("Select Top N", 5, 20, 10)
        top_anomalies = df.sort_values("Overall_Anomaly_Score", ascending=False).head(top_n)

        st.subheader(f"Detailed Analysis of Top {top_n} Anomalous Municipalities:")
        for idx, row in top_anomalies.iterrows():
            muni_name = f"{row.iloc[1]} ({row.iloc[0]})"  # Municipality (District)
            score = round(row["Overall_Anomaly_Score"], 3)

            st.markdown(f"**{idx+1}. {muni_name} - Score: {score}**")

            # Show top 3 most anomalous features for this municipality
            muni_zscores = z_scores.loc[idx]
            top_features = muni_zscores.sort_values(ascending=False).head(3)
            for feat, zval in top_features.items():
                st.write(f"{feat}: {round(features.loc[idx, feat], 1)}% (Z-score: {round(zval, 2)})")
            st.write("---")

    # ---- SEARCH VIEW ----
    elif view_option == "Search Municipality":
        search_term = st.text_input("Enter Municipality or District name:")
        if search_term:
            matches = df[df.iloc[:, 1].str.contains(search_term, case=False, na=False)]  # searching in municipality col
            if not matches.empty:
                for idx, row in matches.iterrows():
                    muni_name = f"{row.iloc[1]} ({row.iloc[0]})"
                    score = round(row["Overall_Anomaly_Score"], 3)
                    st.markdown(f"### {muni_name} - Score: {score}")

                    # Show top 5 anomalous features for this municipality
                    muni_zscores = z_scores.loc[idx]
                    top_features = muni_zscores.sort_values(ascending=False).head(5)
                    for feat, zval in top_features.items():
                        st.write(f"{feat}: {round(features.loc[idx, feat], 1)}% (Z-score: {round(zval, 2)})")
                    st.write("---")
            else:
                st.warning("No matching municipality found. Try again.")
