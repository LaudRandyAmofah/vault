import io
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")
st.title("Cleanup Vault (Optimized)")

# ---------- Helpers & Caching ----------
@st.cache_data(show_spinner=False)
def load_df(uploaded):
    if uploaded.name.endswith(".csv"):
        # Try PyArrow if present; fallback to default
        try:
            return pd.read_csv(uploaded, engine="pyarrow")
        except Exception:
            return pd.read_csv(uploaded)
    else:
        # openpyxl is included in requirements
        return pd.read_excel(uploaded)

@st.cache_data(show_spinner=True)
def summarize_missing(df):
    miss_cnt = df.isna().sum()
    miss_pct = (miss_cnt / len(df) * 100).round(2)
    miss = pd.DataFrame({"column": df.columns,
                         "missing_count": miss_cnt.values,
                         "missing_pct": miss_pct.values})
    return miss

@st.cache_data(show_spinner=True)
def analyze_issues(df, numeric_cols):
    """Return issues_df (long), clean_df (same columns), and index sets."""
    n = len(df)
    # --- Missing rows ---
    missing_mask = df.isnull().any(axis=1)
    missing_idx = set(df.index[missing_mask].tolist())

    # --- Outliers via IQR (vectorized per column into index sets) ---
    outlier_idx = set()
    for col in numeric_cols:
        col_series = df[col]
        Q1, Q3 = col_series.quantile([0.25, 0.75])
        iqr = Q3 - Q1
        if pd.isna(iqr) or iqr == 0:
            continue
        mask = (col_series < Q1 - 1.5 * iqr) | (col_series > Q3 + 1.5 * iqr)
        if mask.any():
            outlier_idx.update(df.index[mask].tolist())

    # --- Anomalies: negative numeric values ---
    negative_idx = set()
    for col in numeric_cols:
        mask = df[col] < 0
        if mask.any():
            negative_idx.update(df.index[mask].tolist())

    # --- Build issues table only once (avoid repeated concatenations) ---
    issues_records = []
    if missing_idx:
        for i in missing_idx:
            row = df.loc[i]
            issues_records.append({**row.to_dict(), "Issue": "Missing value"})
    if outlier_idx:
        # For outliers, store column list for that row to reduce rowsize expansion
        for i in outlier_idx:
            # Identify which columns tripped (optional detail)
            cols = []
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                iqr = Q3 - Q1
                if pd.isna(iqr) or iqr == 0: 
                    continue
                val = df.at[i, col]
                if pd.notna(val) and (val < Q1 - 1.5 * iqr or val > Q3 + 1.5 * iqr):
                    cols.append(col)
            row = df.loc[i]
            issues_records.append({**row.to_dict(), "Issue": f"Outlier in {', '.join(cols) if cols else 'numeric cols'}"})
    if negative_idx:
        for i in negative_idx:
            cols = [c for c in numeric_cols if pd.notna(df.at[i, c]) and df.at[i, c] < 0]
            row = df.loc[i]
            issues_records.append({**row.to_dict(), "Issue": f"Negative value in {', '.join(cols) if cols else 'numeric cols'}"})

    issues_df = pd.DataFrame(issues_records) if issues_records else pd.DataFrame()

    # --- Cleaned data (drop unique indices) ---
    to_drop = missing_idx.union(outlier_idx).union(negative_idx)
    if to_drop:
        clean_df = df.drop(index=list(to_drop))
    else:
        clean_df = df.copy()

    return issues_df, clean_df, (missing_idx, outlier_idx, negative_idx)

def to_csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    buf.write(df.to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    return buf

# ---------- UI ----------
uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded:
    with st.spinner("Loading data..."):
        df = load_df(uploaded)

    st.subheader("Raw Data Preview")
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(200), use_container_width=True)  # show only a slice

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # ----- Visualizations -----
    st.subheader("Quick Visuals")

    cols_viz = st.columns(2)
    with cols_viz[0]:
        # Missing summary chart (fast + scalable)
        miss = summarize_missing(df)
        miss_nonzero = miss[miss["missing_count"] > 0]
        if not miss_nonzero.empty:
            fig_miss = px.bar(
                miss_nonzero.sort_values("missing_pct", ascending=False),
                x="column", y="missing_pct",
                title="Missing Values (%) by Column"
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.write("No missing values detected.")

    with cols_viz[1]:
        # Boxplot on sample for large datasets to avoid heavy rendering
        if numeric_cols:
            if len(df) > 30000:
                samp = df[numeric_cols].sample(30000, random_state=42)
                st.caption("Showing boxplots on a 30k-row sample for performance.")
                fig_box = px.box(samp, y=numeric_cols, points=False)
            else:
                fig_box = px.box(df, y=numeric_cols, points=False)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.write("No numeric columns found.")

    # ----- Analyze button -----
    if st.button("Analyze Issues", type="primary"):
        with st.spinner("Analyzing (cached on repeat runs)..."):
            issues_df, clean_df, _ = analyze_issues(df, numeric_cols)

        st.subheader("Issues Table")
        if not issues_df.empty:
            st.dataframe(issues_df.head(500), use_container_width=True)  # cap rows in UI
            st.download_button(
                "Download Full Issues CSV",
                data=to_csv_bytes(issues_df),
                file_name="issues_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.success("No issues found.")

        st.subheader("Cleaned Data (preview)")
        st.dataframe(clean_df.head(500), use_container_width=True)
        st.download_button(
            "Download Cleaned CSV",
            data=to_csv_bytes(clean_df),
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )
