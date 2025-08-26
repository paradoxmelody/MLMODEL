import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== Load Models =====
iso_forest = joblib.load("C:\\Users\\ICT 2022\\OneDrive - University of the Western Cape\\Desktop\\Deployment\\iso_forest_model.pkl")
lof = joblib.load("C:\\Users\\ICT 2022\\OneDrive - University of the Western Cape\\Desktop\\Deployment\\lof_model.pkl")
scaler = joblib.load("C:\\Users\\ICT 2022\\OneDrive - University of the Western Cape\\Desktop\\Deployment\\scaler.pkl")
protest_model = joblib.load("C:\\Users\\ICT 2022\\OneDrive - University of the Western Cape\\Desktop\\Deployment\\protest_risk_model.pkl")

st.set_page_config(page_title="Risk & Anomaly Dashboard", layout="wide")
st.title("🛡️ Integrated Risk & Anomaly Dashboard")

# Sidebar navigation
menu = st.sidebar.radio("📌 Select Feature:", [
    "Census Anomaly Detection",
    "Protest Risk Prediction",
    "Combined Risk Alerts"
])

# ======================================================
# 1) CENSUS ANOMALY DETECTION (your original feature)
# ======================================================
if menu == "Census Anomaly Detection":
    st.header("📊 Census Anomaly Detection")

    uploaded_file = st.file_uploader("Upload CSV with municipality metrics", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        features = df.iloc[:, 3:]  # assuming first 3 cols = province/municipality
        X_scaled = scaler.transform(features.values)

        # Predictions
        iso_predictions = iso_forest.predict(X_scaled)
        lof_predictions = lof.fit_predict(X_scaled)

        # Z-scores
        z_scores = np.abs((features - features.mean()) / features.std(ddof=0))
        z_score_anomalies = (z_scores > 2.0).any(axis=1)

        # Add results
        df['Isolation_Forest_Score'] = iso_forest.decision_function(X_scaled)
        df['Isolation_Forest_Anomaly'] = iso_predictions
        df['LOF_Anomaly'] = lof_predictions
        df['Z_Score_Anomaly'] = z_score_anomalies

        df['Overall_Anomaly_Score'] = (
            (df['Isolation_Forest_Score'] - df['Isolation_Forest_Score'].min()) /
            (df['Isolation_Forest_Score'].max() - df['Isolation_Forest_Score'].min()) +
            (lof.negative_outlier_factor_ - lof.negative_outlier_factor_.min()) /
            (lof.negative_outlier_factor_.max() - lof.negative_outlier_factor_.min()) +
            z_score_anomalies.astype(int)
        ) / 3

        # Sidebar
        st.sidebar.header("🔎 Explore Anomalies")
        view_option = st.sidebar.radio("Select View:", ["Top Anomalies", "Search Municipality"])

        if view_option == "Top Anomalies":
            top_n = st.sidebar.slider("Select Top N", 5, 20, 10)
            top_anomalies = df.sort_values("Overall_Anomaly_Score", ascending=False).head(top_n)

            st.subheader(f"Detailed Analysis of Top {top_n} Anomalous Municipalities:")
            for idx, row in top_anomalies.iterrows():
                muni_name = f"{row.iloc[1]} ({row.iloc[0]})"
                score = round(row["Overall_Anomaly_Score"], 3)

                st.markdown(f"**{idx+1}. {muni_name} - Score: {score}**")
                muni_zscores = z_scores.loc[idx]
                top_features = muni_zscores.sort_values(ascending=False).head(3)
                for feat, zval in top_features.items():
                    st.write(f"{feat}: {round(features.loc[idx, feat], 1)}% (Z-score: {round(zval, 2)})")
                st.write("---")

        elif view_option == "Search Municipality":
            search_term = st.text_input("Enter Municipality or District name:")
            if search_term:
                matches = df[df.iloc[:, 1].str.contains(search_term, case=False, na=False)]
                if not matches.empty:
                    for idx, row in matches.iterrows():
                        muni_name = f"{row.iloc[1]} ({row.iloc[0]})"
                        score = round(row["Overall_Anomaly_Score"], 3)
                        st.markdown(f"### {muni_name} - Score: {score}")
                        muni_zscores = z_scores.loc[idx]
                        top_features = muni_zscores.sort_values(ascending=False).head(5)
                        for feat, zval in top_features.items():
                            st.write(f"{feat}: {round(features.loc[idx, feat], 1)}% (Z-score: {round(zval, 2)})")
                        st.write("---")
                else:
                    st.warning("No matching municipality found. Try again.")

# ======================================================
# 2) PROTEST RISK PREDICTION
# ======================================================
elif menu == "Protest Risk Prediction":
    st.header("🔥 Protest Risk Prediction (Logistic Regression)")

    uploaded_file = st.file_uploader("Upload crime dataset", type=["csv"], key="crime_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Predict with saved pipeline
        df["Predicted_Probability"] = protest_model.predict_proba(df)[:, 1]
        df["Predicted_Class"] = protest_model.predict(df)

        # Format percentage
        df["Predicted_Probability"] = (df["Predicted_Probability"] * 100).round(1).astype(str) + "%"

        # Sidebar province grouping
        provinces = df["Province"].unique()
        selected_province = st.sidebar.selectbox("Filter by Province", ["All"] + list(provinces))

        if selected_province != "All":
            df = df[df["Province"] == selected_province]

        # Show top 20 by probability
        top20 = df.sort_values("Predicted_Probability", ascending=False).head(20)
        st.subheader("Top 20 High-Risk Predictions")
        st.dataframe(top20)

# ======================================================
# 3) COMBINED RISK ALERTS
# ======================================================
elif menu == "Combined Risk Alerts":
    st.header("🚨 Combined Risk Alerts")

    census_file = st.file_uploader("Upload Census Anomaly Results CSV", type=["csv"], key="census_upload")
    protest_file = st.file_uploader("Upload Protest Predictions CSV", type=["csv"], key="protest_upload")

    if census_file and protest_file:
        census_df = pd.read_csv(census_file)
        protest_df = pd.read_csv(protest_file)

        # Merge on Province + District
        merged = protest_df.merge(
            census_df[["Province", "District", "Overall_Anomaly_Score"]],
            on=["Province", "District"], how="left"
        )

        # Risk index = combination
        merged["Risk_Index"] = (
            merged["Overall_Anomaly_Score"].fillna(0) +
            merged["Predicted_Probability"].str.replace("%", "").astype(float) / 100
        ) / 2

        # Find most at risk
        top_risk = merged.sort_values("Risk_Index", ascending=False).head(1)
        st.subheader("⚠️ Most At-Risk Location")
        st.dataframe(top_risk)

        st.success(
            f"🚨 ALERT: {top_risk.iloc[0]['District']} in {top_risk.iloc[0]['Province']} "
            f"is the MOST at risk with combined score {round(top_risk.iloc[0]['Risk_Index'], 3)}"
        )
