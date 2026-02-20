import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Cleanup Vault")

st.markdown("Upload CSV or Excel files to detect issues and get a cleaned version.")

# ────────────────────────────────────────────────
# File uploader
# ────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded is not None:
    with st.spinner("Loading file..."):
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

    # Basic info
    st.subheader("File Info")
    st.write(f"Rows: **{len(df):,}** | Columns: **{len(df.columns)}**")

    # Show preview (limited rows)
    st.subheader("Data Preview (first 500 rows)")
    st.dataframe(df.head(500))

    # ────────────────────────────────────────────────
    # Cached sampling function for visualizations
    # ────────────────────────────────────────────────
    @st.cache_data(show_spinner="Preparing visualization sample...")
    def get_sample_for_viz(_df, max_rows=20000):
        if len(_df) > max_rows:
            return _df.sample(max_rows, random_state=42)
        return _df

    # ────────────────────────────────────────────────
    # Boxplot – only on sample
    # ────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if numeric_cols:
        st.subheader("Boxplots of numeric columns (sampled data)")
        sample_df = get_sample_for_viz(df)
        fig_box = px.box(sample_df, y=numeric_cols, points="outliers", title="Boxplot – sample of up to 20,000 rows")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns found → no boxplots available.")

    # ────────────────────────────────────────────────
    # Analyze button + heavy computation
    # ────────────────────────────────────────────────
    if st.button("Analyze Issues & Clean Data", type="primary"):
        with st.spinner("Analyzing data... this may take a while on large files"):
            # Missing values
            missing_rows = df[df.isnull().any(axis=1)].copy()
            if not missing_rows.empty:
                missing_rows['Issue'] = 'Missing value'

            # Outliers (IQR method)
            outliers = pd.DataFrame()
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                iqr = Q3 - Q1
                mask = (df[col] < Q1 - 1.5 * iqr) | (df[col] > Q3 + 1.5 * iqr)
                temp = df[mask].copy()
                if not temp.empty:
                    temp['Issue'] = f'Outlier in {col}'
                    outliers = pd.concat([outliers, temp], ignore_index=True)

            # Negative values in numeric columns
            anomalies = pd.DataFrame()
            for col in numeric_cols:
                mask = df[col] < 0
                temp = df[mask].copy()
                if not temp.empty:
                    temp['Issue'] = f'Negative value in {col}'
                    anomalies = pd.concat([anomalies, temp], ignore_index=True)

            # Combine all issues
            dfs_to_concat = [df_part for df_part in [missing_rows, outliers, anomalies] if not df_part.empty]
            if dfs_to_concat:
                issues_df = pd.concat(dfs_to_concat, ignore_index=True)
            else:
                issues_df = pd.DataFrame()

            # Create cleaned version
            clean_df = df.copy()
            if not missing_rows.empty:
                clean_df = clean_df.drop(missing_rows.index, errors='ignore')
            if not outliers.empty or not anomalies.empty:
                bad_idx = pd.concat([outliers, anomalies]).index
                clean_df = clean_df.drop(bad_idx, errors='ignore')

        # ────────────────────────────────────────────────
        # Results
        # ────────────────────────────────────────────────
        st.success("Analysis complete!")

        st.subheader("Detected Issues")
        if issues_df.empty:
            st.success("No issues found!")
        else:
            st.write(f"Found **{len(issues_df):,}** problematic rows")
            # Show only head + option to download full
            st.dataframe(issues_df.head(1000))
            if len(issues_df) > 1000:
                st.info(f"Showing first 1,000 rows of {len(issues_df):,} total issues")

        st.subheader("Cleaned Data Preview")
        st.dataframe(clean_df.head(500))
        if len(clean_df) < len(df):
            st.write(f"Rows reduced from **{len(df):,}** → **{len(clean_df):,}**")

        # ────────────────────────────────────────────────
        # Download buttons
        # ────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Cleaned CSV",
                data=clean_df.to_csv(index=False).encode('utf-8'),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        with col2:
            if not issues_df.empty:
                st.download_button(
                    label="Download Issues Report CSV",
                    data=issues_df.to_csv(index=False).encode('utf-8'),
                    file_name="issues_report.csv",
                    mime="text/csv"
                )
