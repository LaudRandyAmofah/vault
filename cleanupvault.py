###Creator: Laud Randy Amofah
###Datestamp: February 20, 2026 11:28 (24hr)
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Cleanup Vault")

# File uploader
uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
if uploaded:
    # Read file based on type
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Visualizations
    st.subheader("Visualizations")

    # Missing values heatmap
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.isnull(), cbar=False)
    st.pyplot(fig)

    # Boxplots for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        fig2 = px.box(df, y=numeric_cols, points="all")
        st.plotly_chart(fig2)

    # Analyze Issues
    if st.button("Analyze Issues"):
        # --- Missing values ---
        missing_rows = df[df.isnull().any(axis=1)].copy()
        if not missing_rows.empty:
            missing_rows['Issue'] = 'Missing value'

        # --- Outliers using IQR ---
        outliers = pd.DataFrame()
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            iqr = Q3 - Q1
            mask = (df[col] < Q1 - 1.5 * iqr) | (df[col] > Q3 + 1.5 * iqr)
            temp = df[mask].copy()
            temp['Issue'] = f'Outlier in {col}'
            outliers = pd.concat([outliers, temp], ignore_index=True)

        # --- Anomalies: example negative numeric values ---
        anomalies = pd.DataFrame()
        for col in numeric_cols:
            mask = df[col] < 0
            temp = df[mask].copy()
            temp['Issue'] = f'Negative value in {col}'
            anomalies = pd.concat([anomalies, temp], ignore_index=True)

        # --- Combine all issues ---
        dfs_to_concat = []
        if not missing_rows.empty:
            dfs_to_concat.append(missing_rows)
        if not outliers.empty:
            dfs_to_concat.append(outliers)
        if not anomalies.empty:
            dfs_to_concat.append(anomalies)

        if dfs_to_concat:
            issues_df = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            issues_df = pd.DataFrame()
        
        st.subheader("Issues Table")
        if not issues_df.empty:
            st.dataframe(issues_df)
        else:
            st.write("No issues found!")

        # --- Cleaned data: example dropping missing + outliers + anomalies ---
        clean_df = df.copy()
        # Drop missing rows
        clean_df = clean_df.drop(missing_rows.index, errors='ignore')
        # Drop outliers and anomalies
        clean_df = clean_df.drop(pd.concat([outliers, anomalies]).index, errors='ignore')

        st.subheader("Cleaned Data")
        st.dataframe(clean_df)

        # --- Download buttons ---
        st.download_button(
            "Download Cleaned CSV",
            clean_df.to_csv(index=False),
            "cleaned_data.csv"
        )
        st.download_button(
            "Download Issues CSV",
            issues_df.to_csv(index=False),
            "issues_report.csv"
        )
