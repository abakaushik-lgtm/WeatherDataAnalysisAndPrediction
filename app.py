import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Set page configuration
st.set_page_config(
    page_title="Weather Analytics & ML Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and responsive elements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Font styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Center title styling - Compact Hero Banner */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 18px 20px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
    }
    
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }
    
    .app-header p {
        font-size: 0.95rem;
        font-weight: 300;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    
    /* KPI Card Container styling */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    /* Individual KPI Card */
    .kpi-card {
        flex: 1;
        min-width: 150px;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        text-align: center;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        border-color: #3b82f6;
    }
    
    .kpi-icon {
        font-size: 1.6rem;
        margin-bottom: 5px;
        display: flex;
        justify-content: center;
    }
    
    .kpi-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 3px;
        text-align: center;
    }
    
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-color);
        text-align: center;
        display: block;
        margin: 0 auto;
    }
    
    /* Prediction output card */
    .prediction-container {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(96, 165, 250, 0.02) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 20px;
        padding: 25px;
        margin-top: 15px;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.08);
        text-align: center;
    }
    
    .prediction-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 5px;
    }
    
    .prediction-val {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        line-height: 1.1;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 6px 14px;
        background-color: #10b981;
        color: white;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 3px 8px rgba(16, 185, 129, 0.15);
    }
    
    .confidence-low {
        background-color: #ef4444;
        box-shadow: 0 3px 8px rgba(239, 68, 68, 0.15);
    }
    
    .confidence-medium {
        background-color: #f59e0b;
        box-shadow: 0 3px 8px rgba(245, 158, 11, 0.15);
    }
    
    /* Subtext under prediction */
    .prediction-sub {
        font-size: 0.8rem;
        opacity: 0.6;
        margin-top: 12px;
    }

    /* Details alignment inside prediction container */
    .pred-detail-box {
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
    }
    
    .pred-detail-item {
        text-align: center;
    }
    
    .pred-detail-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.6;
        margin-bottom: 2px;
    }
    
    .pred-detail-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-color);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.7;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Application paths
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_DATASET_DIR, "sample_weather.csv")

# Make sure directory structures exist
os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)

# Generate sample dataset if it doesn't exist
if not os.path.exists(DEFAULT_DATASET_PATH):
    with st.spinner("Generating sample weather dataset..."):
        utils.generate_synthetic_weather_data(DEFAULT_DATASET_PATH)

# Initialize Session State
if "raw_df" not in st.session_state:
    try:
        st.session_state.raw_df = pd.read_csv(DEFAULT_DATASET_PATH)
    except Exception as e:
        st.session_state.raw_df = None
        st.error(f"Error loading default dataset: {e}")

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "is_cleaned" not in st.session_state:
    st.session_state.is_cleaned = False

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = "N/A"

if "model_trained" not in st.session_state:
    # Check if models already exist in filesystem
    model, scaler, metadata, _ = utils.load_best_model()
    if model is not None:
        st.session_state.model_trained = True
        st.session_state.best_model_name = metadata["best_model_name"]
        st.session_state.metrics = pd.DataFrame(metadata["metrics"])
    else:
        st.session_state.model_trained = False
        st.session_state.best_model_name = None
        st.session_state.metrics = None

# Header Banner
st.markdown("""
<div class="app-header">
    <h1>🌦️ Weather Analytics & Temperature Prediction</h1>
    <p>Upload historical weather data, run interactive Exploratory Data Analysis, and train Machine Learning models to predict temperature.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation and Configuration
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Select Panel",
    ["🏠 Dashboard Overview", "📊 Exploratory Data Analysis", "🤖 Train ML Models", "🔮 Predict Temperature", "ℹ️ Project Info"]
)

# Active dataset check
if st.session_state.raw_df is not None:
    df_to_use = st.session_state.cleaned_df if st.session_state.is_cleaned else st.session_state.raw_df
else:
    df_to_use = None

# Helper to check if df has missing values
def has_missing_or_duplicates(df):
    if df is None:
        return False, False
    dups = df.duplicated().sum() > 0
    nulls = df.isnull().sum().sum() > 0
    return dups, nulls

# --- 1. DASHBOARD OVERVIEW PAGE ---
if page == "🏠 Dashboard Overview":
    st.header("🏠 Dashboard Overview")
    
    col_u1, col_u2 = st.columns([2, 1])
    
    with col_u1:
        st.subheader("📁 Upload Weather Dataset")
        uploaded_file = st.file_uploader("Upload your weather dataset CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                # Verify required columns
                required_cols = ["Date", "Temperature", "Humidity", "Wind Speed", "Pressure", "Rainfall"]
                missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
                
                if len(missing_cols) == 0:
                    st.session_state.raw_df = uploaded_df
                    st.session_state.cleaned_df = None
                    st.session_state.is_cleaned = False
                    st.success("Successfully uploaded weather dataset!")
                    st.rerun()
                else:
                    st.error(f"Invalid dataset structure. Missing columns: {', '.join(missing_cols)}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        else:
            st.info("💡 Currently using the **Default Sample Weather Dataset**. You can upload your own weather CSV above.")
            
    with col_u2:
        st.subheader("🧹 Data Status & Cleaning")
        if st.session_state.raw_df is not None:
            dups, nulls = has_missing_or_duplicates(st.session_state.raw_df)
            
            if st.session_state.is_cleaned:
                st.success("✅ Dataset is Cleaned and Imputed.")
                if st.button("Reset to Raw Data", use_container_width=True):
                    st.session_state.cleaned_df = None
                    st.session_state.is_cleaned = False
                    st.rerun()
            else:
                if dups or nulls:
                    st.warning("⚠️ Raw data contains duplicate rows or missing values.")
                    if st.button("🧼 Run Auto-Clean & Impute", type="primary", use_container_width=True):
                        cleaned, d_cnt, m_cnt, cols_m = utils.clean_weather_data(st.session_state.raw_df)
                        st.session_state.cleaned_df = cleaned
                        st.session_state.is_cleaned = True
                        st.session_state.clean_log = {
                            "duplicates": d_cnt,
                            "missing": m_cnt,
                            "by_column": cols_m
                        }
                        st.rerun()
                else:
                    st.success("✨ Raw data contains no duplicates or missing values.")
        else:
            st.write("No dataset loaded.")

    st.markdown("---")
    
    if df_to_use is not None:
        # Calculate Project Metrics for top KPI Cards
        dataset_rows = df_to_use.shape[0]
        dataset_cols = df_to_use.shape[1]
        current_nulls = df_to_use.isnull().sum().sum()
        current_dups = df_to_use.duplicated().sum()
        
        best_r2_text = "N/A"
        if st.session_state.model_trained:
            _, _, metadata, _ = utils.load_best_model()
            if metadata is not None:
                best_r2_text = f"{metadata['r2_score'] * 100:.1f}%"
                
        last_pred_text = st.session_state.last_prediction
        
        # Display KPI Cards
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-title">Dataset Rows</div>
                <div class="kpi-value">{dataset_rows:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📋</div>
                <div class="kpi-title">Dataset Columns</div>
                <div class="kpi-value">{dataset_cols}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔍</div>
                <div class="kpi-title">Missing Values</div>
                <div class="kpi-value">{current_nulls}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🧼</div>
                <div class="kpi-title">Duplicate Rows</div>
                <div class="kpi-value">{current_dups}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-title">Model Accuracy</div>
                <div class="kpi-value">{best_r2_text}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔮</div>
                <div class="kpi-title">Last Prediction</div>
                <div class="kpi-value">{last_pred_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cleaning Results Log
        if st.session_state.is_cleaned and hasattr(st.session_state, "clean_log"):
            log = st.session_state.clean_log
            with st.expander("📝 Show Cleaning Log Details"):
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown(f"**Duplicate Rows Removed:** `{log['duplicates']}`")
                    st.markdown(f"**Total Missing Values Imputed:** `{log['missing']}`")
                with col_c2:
                    st.markdown("**Imputed Counts by Variable:**")
                    for col, cnt in log['by_column'].items():
                        st.markdown(f"- {col}: `{cnt}` values imputed using Median")
                        
        # Dataset Overview
        st.subheader("📋 Dataset Preview")
        col_s1, col_s2 = st.columns([3, 1])
        
        with col_s1:
            st.dataframe(df_to_use.head(10), use_container_width=True)
        with col_s2:
            st.markdown("**Dataset Shape:**")
            st.code(f"Rows: {df_to_use.shape[0]}\nColumns: {df_to_use.shape[1]}")
            
            missing_any = df_to_use.isnull().sum()
            st.markdown("**Missing Values:**")
            st.dataframe(missing_any[missing_any > 0] if missing_any.sum() > 0 else "None", use_container_width=True)

        st.dataframe(df_to_use.describe().T, use_container_width=True)
        
        # Machine Learning Status
        if st.session_state.model_trained:
            st.markdown("---")
            st.subheader("🧠 Machine Learning Model Status")
            
            # Retrieve metrics
            _, _, metadata, _ = utils.load_best_model()
            if metadata is not None:
                best_model = metadata["best_model_name"]
                best_r2 = metadata["r2_score"]
                # Find best model row in metrics
                best_metrics = [m for m in metadata["metrics"] if m["Model"] == best_model][0]
                best_rmse = best_metrics["RMSE"]
                best_mae = best_metrics["MAE"]
                
                # Display 4 cards for Best Model | R2 Score | RMSE | MAE
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 4px solid #10b981;">
                        <div class="kpi-icon">🏆</div>
                        <div class="kpi-title">Best Model</div>
                        <div class="kpi-value" style="font-size: 1.1rem; padding-top: 5px;">{best_model}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c2:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
                        <div class="kpi-icon">🎯</div>
                        <div class="kpi-title">R² Score</div>
                        <div class="kpi-value">{best_r2:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c3:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
                        <div class="kpi-icon">📉</div>
                        <div class="kpi-title">RMSE</div>
                        <div class="kpi-value">{best_rmse:.2f}°C</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c4:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 4px solid #ef4444;">
                        <div class="kpi-icon">📉</div>
                        <div class="kpi-title">MAE</div>
                        <div class="kpi-value">{best_mae:.2f}°C</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📊 Model Performance Comparison")
                metrics_df = pd.DataFrame(metadata["metrics"])
                
                def highlight_best(row):
                    if row["Model"] == best_model:
                        return ["background-color: rgba(16, 185, 129, 0.15); border: 1.5px solid #10b981"] * len(row)
                    return [""] * len(row)
                
                styled_metrics = metrics_df.style.apply(highlight_best, axis=1)
                st.dataframe(styled_metrics, use_container_width=True)
        
    else:
        st.warning("Please upload a dataset or ensure the default dataset is generated to display contents.")

# --- 2. EXPLORATORY DATA ANALYSIS (EDA) PAGE ---
elif page == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    if df_to_use is None:
        st.warning("Please load a dataset on the Dashboard page to analyze.")
    else:
        # Pre-process Dates for seasonal plotting
        df_plot = df_to_use.copy()
        df_plot["Date"] = pd.to_datetime(df_plot["Date"])
        df_plot["Month"] = df_plot["Date"].dt.strftime("%b")
        df_plot["MonthNum"] = df_plot["Date"].dt.month
        df_plot = df_plot.sort_values("MonthNum")
        
        # Navigation tabs inside EDA
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["🕒 Trends & Time Series", "📊 Distributions & Ranges", "🔗 Relationships & Correlations"])
        
        with eda_tab1:
            st.subheader("Weather Parameters Over Time")
            col_t1, col_t2 = st.columns([3, 1])
            with col_t2:
                time_param = st.selectbox(
                    "Select Weather Variable",
                    ["Temperature", "Humidity", "Wind Speed", "Pressure", "Rainfall"]
                )
                ma_window = st.slider("Moving Average Window (Days)", min_value=1, max_value=30, value=7)
                
            with col_t1:
                # Plotly Time Series with Moving Average
                df_ts = df_to_use.copy()
                df_ts["Date"] = pd.to_datetime(df_ts["Date"])
                df_ts = df_ts.sort_values("Date")
                
                # Compute Moving Average
                df_ts["MA"] = df_ts[time_param].rolling(window=ma_window, min_periods=1).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ts["Date"], y=df_ts[time_param], mode='lines', name='Daily Value', opacity=0.4, line=dict(color='#93c5fd')))
                fig.add_trace(go.Scatter(x=df_ts["Date"], y=df_ts["MA"], mode='lines', name=f'{ma_window}-Day MA', line=dict(color='#1d4ed8', width=2.5)))
                
                fig.update_layout(
                    title=f"{time_param} Trend Over Time",
                    xaxis_title="Date",
                    yaxis_title=time_param,
                    hovermode="x unified",
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Monthly Averages
            st.markdown("---")
            st.subheader("Average Monthly Weather Trends")
            
            monthly_avg = df_plot.groupby(["MonthNum", "Month"]).mean(numeric_only=True).reset_index()
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                fig_m1 = px.line(
                    monthly_avg, x="Month", y="Temperature",
                    title="Monthly Average Temperature Trend",
                    markers=True, line_shape="spline",
                    color_discrete_sequence=["#ef4444"]
                )
                fig_m1.update_layout(template="plotly_white")
                st.plotly_chart(fig_m1, use_container_width=True)
                
            with col_m2:
                fig_m2 = px.bar(
                    monthly_avg, x="Month", y="Rainfall",
                    title="Monthly Average Rainfall Distribution",
                    color_discrete_sequence=["#3b82f6"]
                )
                fig_m2.update_layout(template="plotly_white")
                st.plotly_chart(fig_m2, use_container_width=True)

        with eda_tab2:
            st.subheader("Data Distributions and Variabilities")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                dist_col = st.selectbox(
                    "Histogram Feature",
                    ["Temperature", "Humidity", "Wind Speed", "Pressure", "Rainfall"]
                )
                fig_hist = px.histogram(
                    df_to_use, x=dist_col,
                    title=f"Distribution of {dist_col}",
                    color_discrete_sequence=["#3b82f6"],
                    marginal="box",
                    nbins=35
                )
                fig_hist.update_layout(template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col_d2:
                box_col = st.selectbox(
                    "Boxplot Parameter (Grouped by Month)",
                    ["Temperature", "Humidity", "Wind Speed", "Pressure"]
                )
                fig_box = px.box(
                    df_plot, x="Month", y=box_col,
                    title=f"Monthly Variability of {box_col}",
                    color="Month",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_box.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

        with eda_tab3:
            st.subheader("Variable Relationships & Feature Correlations")
            
            col_r1, col_r2 = st.columns([2, 1])
            
            with col_r2:
                scatter_x = st.selectbox("X-Axis Feature", ["Humidity", "Wind Speed", "Pressure", "Rainfall"], index=0)
                scatter_y = st.selectbox("Y-Axis Feature (Target)", ["Temperature"], index=0)
                scatter_color = st.selectbox("Color Code Parameter", ["Rainfall", "Humidity", "Wind Speed", "Pressure"], index=0)
                
            with col_r1:
                fig_scat = px.scatter(
                    df_to_use, x=scatter_x, y=scatter_y, color=scatter_color,
                    title=f"Scatter Plot: {scatter_x} vs {scatter_y}",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    opacity=0.7,
                    trendline="ols" if df_to_use[scatter_x].nunique() > 1 else None,
                    trendline_color_override="#ef4444"
                )
                fig_scat.update_layout(template="plotly_white")
                st.plotly_chart(fig_scat, use_container_width=True)
                
            # Heatmap
            st.markdown("---")
            st.subheader("Correlation Matrix Heatmap")
            
            col_h1, col_h2 = st.columns([1, 2])
            
            with col_h1:
                st.markdown("""
                **Analyzing the Correlation Matrix:**
                * **1.0** indicates perfect positive linear correlation.
                * **-1.0** indicates perfect negative linear correlation.
                * **0.0** indicates no linear relationship.
                
                Observe how temperature relates to humidity, pressure, and wind speed to verify the physics of the model features.
                """)
                
            with col_h2:
                numeric_df = df_to_use.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                
                fig, ax = plt.subplots(figsize=(7, 5))
                # Custom color palette matching the blue theme
                sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    cmap="Blues", 
                    fmt=".2f", 
                    linewidths=0.5,
                    square=True,
                    cbar_kws={"shrink": .8},
                    ax=ax
                )
                plt.title("Weather Features Correlation Heatmap", fontsize=12, pad=15)
                # Apply outfit font styling to plot ticks
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# --- 3. TRAIN ML MODELS PAGE ---
elif page == "🤖 Train ML Models":
    st.header("🤖 Train and Compare ML Regressors")
    
    if df_to_use is None:
        st.warning("Please load a dataset on the Dashboard page to train models.")
    else:
        # Check if dataset has missing values or duplicates
        dups, nulls = has_missing_or_duplicates(df_to_use)
        if dups or nulls:
            st.warning("⚠️ The current dataset contains missing values or duplicates. Cleaning the dataset is highly recommended before training.")
            if st.button("🧹 Click to Auto-Clean & Impute Now"):
                cleaned, d_cnt, m_cnt, cols_m = utils.clean_weather_data(st.session_state.raw_df)
                st.session_state.cleaned_df = cleaned
                st.session_state.is_cleaned = True
                st.session_state.clean_log = {"duplicates": d_cnt, "missing": m_cnt, "by_column": cols_m}
                st.rerun()
        
        st.markdown("Configure the hyper-parameters and split ratios for training regression models below.")
        
        col_tr1, col_tr2 = st.columns([1, 2])
        
        with col_tr1:
            st.subheader("⚙️ Settings")
            split_ratio = st.slider("Train / Test Split Ratio (%)", min_value=50, max_value=90, value=80, step=5)
            test_size = (100 - split_ratio) / 100.0
            
            train_button = st.button("🚀 Train Regression Models", type="primary", use_container_width=True)
            
        with col_tr2:
            st.subheader("💡 Selected Features")
            st.info("🎯 **Target Column:** `Temperature` (°C)")
            st.info("📋 **Input Features:** `Humidity`, `Wind Speed`, `Pressure`, `Rainfall`")
            
        if train_button:
            with st.spinner("Training Linear Regression, Random Forest, and Decision Tree..."):
                trained_models, scaler, metrics_df, best_model_name, feature_names = utils.train_and_evaluate_models(
                    df_to_use, test_size=test_size
                )
                st.session_state.model_trained = True
                st.session_state.best_model_name = best_model_name
                st.session_state.metrics = metrics_df
                
                st.success(f"🎉 Successfully trained all models! **{best_model_name}** was identified as the best performing model.")
        
        # Display Results
        if st.session_state.model_trained and st.session_state.metrics is not None:
            st.markdown("---")
            st.subheader("📊 Regression Model Performance Comparison")
            
            # Highlight best model based on R2 Score
            def highlight_best(row):
                if row["Model"] == st.session_state.best_model_name:
                    return ["background-color: rgba(16, 185, 129, 0.15); border: 1.5px solid #10b981"] * len(row)
                return [""] * len(row)
            
            styled_metrics = st.session_state.metrics.style.apply(highlight_best, axis=1)
            st.dataframe(styled_metrics, use_container_width=True)
            
            st.success(f"🏆 **Best Performing Model:** {st.session_state.best_model_name} (Highest R² Score: **{st.session_state.metrics.loc[st.session_state.metrics['Model'] == st.session_state.best_model_name, 'R² Score'].values[0]:.4f}**)")
            
            # Feature Importance for Random Forest
            # Load metadata to fetch feature importance
            model, _, _, _ = utils.load_best_model()
            if model is not None:
                st.markdown("---")
                st.subheader("🌲 Feature Importance (Random Forest Regressor)")
                
                # Retrieve best Random Forest model (or load it specifically for importance)
                rf_model = None
                if st.session_state.best_model_name == "Random Forest Regressor":
                    rf_model = model
                else:
                    # Let's train/retrieve RF from models to show feature importances
                    # Or check if saved
                    try:
                        # Re-run or extract from trained models if we didn't save it
                        # Since we only saved best_model.pkl, let's load or build importances
                        # It is easiest to fit a quick Random Forest on the fly if it wasn't the best model
                        # or display a message
                        pass
                    except:
                        pass
                
                # To guarantee we have RF importances:
                try:
                    # Train a quick RF model if not already active to show the graph
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    X_cols = ["Humidity", "Wind Speed", "Pressure", "Rainfall"]
                    y_col = "Temperature"
                    temp_df = df_to_use.dropna(subset=X_cols + [y_col])
                    
                    scaler_temp = StandardScaler()
                    X_sc = scaler_temp.fit_transform(temp_df[X_cols])
                    rf.fit(X_sc, temp_df[y_col])
                    
                    importances = rf.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    importance_df = pd.DataFrame({
                        "Feature": [X_cols[i] for i in indices],
                        "Importance": [importances[i] for i in indices]
                    })
                    
                    fig_imp = px.bar(
                        importance_df, 
                        x="Importance", 
                        y="Feature", 
                        orientation='h',
                        title="Relative Impact of Weather Variables on Temperature",
                        color="Importance",
                        color_continuous_scale="Blues",
                        text_auto='.3f'
                    )
                    fig_imp.update_layout(
                        template="plotly_white",
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as ex:
                    st.error(f"Could not compute feature importances: {ex}")
        else:
            st.info("ℹ️ Models have not been trained yet. Click the 'Train Regression Models' button above to start.")

# --- 4. PREDICT TEMPERATURE PAGE ---
elif page == "🔮 Predict Temperature":
    st.header("🔮 Predict Future Temperature")
    
    # Load model and scaler
    model, scaler, metadata, x_train_vals = utils.load_best_model()
    
    if model is None:
        st.warning("⚠️ No trained model found! Please go to the 'Train ML Models' tab to train and save the best model first.")
    else:
        st.markdown(f"Using the saved best model: **{metadata['best_model_name']}** (R² Score: **{metadata['r2_score']:.4f}**)")
        
        col_p1, col_p2 = st.columns([1, 1])
        
        with col_p1:
            st.subheader("📥 Input Current Weather Parameters")
            
            # Interactive Sliders with sensible boundaries
            # Default values are set near the average values in synthetic generation
            humidity_input = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
            wind_input = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=120.0, value=15.0, step=0.5)
            pressure_input = st.slider("Atmospheric Pressure (hPa)", min_value=950.0, max_value=1060.0, value=1013.25, step=0.5)
            rainfall_input = st.slider("Rainfall (mm)", min_value=0.0, max_value=150.0, value=0.0, step=0.1)
            
            predict_click = st.button("🔮 Calculate Predicted Temperature", type="primary", use_container_width=True)
            
        with col_p2:
            st.subheader("🎯 Prediction Output")
            
            if predict_click:
                # Prepare inputs for prediction
                input_df = pd.DataFrame(
                    [[humidity_input, wind_input, pressure_input, rainfall_input]], 
                    columns=metadata["feature_names"]
                )
                
                # Scale input vector
                scaled_inputs = scaler.transform(input_df)
                
                # Predict
                predicted_temp = model.predict(scaled_inputs)[0]
                
                # Determine weather condition
                condition = utils.determine_weather_condition(predicted_temp, humidity_input, rainfall_input)
                
                # Calculate Confidence Score
                confidence = utils.calculate_confidence_score(
                    [humidity_input, wind_input, pressure_input, rainfall_input],
                    x_train_vals,
                    metadata["r2_score"]
                )
                
                # Update Session State Predictions History
                history_entry = {
                    "Temperature": f"{predicted_temp:.2f}°C",
                    "Humidity": f"{humidity_input:.1f}%",
                    "Wind": f"{wind_input:.1f} km/h",
                    "Pressure": f"{pressure_input:.1f} hPa",
                    "Rainfall": f"{rainfall_input:.1f} mm",
                    "Predicted Temp": f"{predicted_temp:.2f}°C",
                    "Weather Condition": condition,
                    "Confidence": f"{confidence}%",
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.prediction_history.append(history_entry)
                st.session_state.last_prediction = f"{predicted_temp:.1f}°C"
                
                # Style confidence badge class
                conf_class = "confidence-badge"
                if confidence < 50.0:
                    conf_class += " confidence-low"
                elif confidence < 75.0:
                    conf_class += " confidence-medium"
                
                # Display beautiful prediction card - Improved Layout
                st.markdown(f"""
                <div class="prediction-container">
                    <div style="margin-bottom: 12px;">
                        <div class="prediction-label" style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 2px;">Predicted Temperature</div>
                        <div class="prediction-val" style="font-size: 3rem; font-weight: 800; margin: 0;">🌡️ {predicted_temp:.1f}°C</div>
                    </div>
                    <div class="pred-detail-box">
                        <div class="pred-detail-item">
                            <div class="pred-detail-label">Weather Condition</div>
                            <div class="pred-detail-value">{condition}</div>
                        </div>
                        <div style="border-left: 1px solid rgba(128, 128, 128, 0.15); height: 35px; display: inline-block;"></div>
                        <div class="pred-detail-item">
                            <div class="pred-detail-label">Confidence</div>
                            <div class="{conf_class}" style="margin: 0; font-size: 0.9rem; padding: 4px 10px;">{confidence}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge representation of temperature
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(predicted_temp, 2),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Temperature Indicator (°C)", 'font': {'size': 16, 'family': 'Outfit'}},
                    gauge={
                        'axis': {'range': [-10, 45], 'tickwidth': 1, 'tickcolor': "gray"},
                        'bar': {'color': "#3b82f6"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [-10, 5], 'color': '#dbeafe'}, # cold
                            {'range': [5, 18], 'color': '#e0f2fe'},  # cool
                            {'range': [18, 28], 'color': '#fef3c7'}, # warm
                            {'range': [28, 45], 'color': '#fee2e2'}  # hot
                        ]
                    }
                ))
                fig_g.update_layout(
                    height=240,
                    margin=dict(l=20, r=20, t=30, b=10),
                    template="plotly_white"
                )
                st.plotly_chart(fig_g, use_container_width=True)
                
            else:
                st.info("Click the 'Calculate Predicted Temperature' button to display the estimate.")
        
        # Display Prediction History Log below the columns (full width)
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.subheader("📋 Session Prediction History")
        if len(st.session_state.prediction_history) > 0:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # Show history dataframe
            st.dataframe(history_df, use_container_width=True)
            
            # Export all prediction history
            history_csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Prediction History as CSV",
                data=history_csv,
                file_name="weather_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No predictions made in this session yet.")

# --- 5. PROJECT INFO PAGE ---
elif page == "ℹ️ Project Info":
    st.header("ℹ️ Weather Analytics Project Information")
    
    st.markdown(r"""
    ### 🌦️ Project Overview
    This Weather Data Analysis and Prediction application provides interactive dashboards and machine learning models for analyzing and forecasting climate conditions.
    It demonstrates a complete end-to-end data science workflow:
    1. **Data Ingestion**: Support for uploading and parsing custom CSV weather datasets.
    2. **Data Cleaning**: Automating duplicate detection and missing value imputation (median imputation).
    3. **Exploratory Data Analysis (EDA)**: Interactive data visualizations showing seasonal weather patterns, histograms, box plots, scatter plots, and correlation matrices.
    4. **Feature Engineering & Preprocessing**: Splitting datasets, feature standardization, and loading ML models.
    5. **Machine Learning Regressors**: Parallel training and metrics-based comparison of:
       * *Linear Regression* (parametric baseline)
       * *Decision Tree Regressor* (non-parametric rule-based model)
       * *Random Forest Regressor* (ensemble learning method)
    6. **Dynamic Inference**: Custom prediction portal evaluating forecasting confidence based on statistical data distance from the training distribution.

    ---

    ### 🏗️ Directory Structure
    ```text
    Weather-Prediction/
    │── app.py                   # Main Streamlit UI and dashboard logic
    │── utils.py                 # Backend algorithms (cleaning, training, and metrics)
    │── requirements.txt         # Required library configurations
    │── README.md                # Markdown setup manual
    │── dataset/                 # Data repository
    │   └── sample_weather.csv   # Auto-generated 1000-day realistic weather data
    └── models/                  # Serialized pipelines and metadata
        ├── best_model.pkl       # Fitted predictor (joblib serialized)
        ├── scaler.pkl           # Feature scaler pipeline (joblib serialized)
        ├── metadata.pkl         # Validation benchmarks and configuration dict
        └── x_train_vals.pkl     # Reference training limits (for confidence scoring)
    ```

    ---

    ### 🛠️ Technical Details
    * **Confidence Index Formula**:
      $$\text{Confidence} = \max(10, R^2 \times 100) - \text{Penalty}$$
      Where the penalty represents structural extrapolation weight:
      $$\text{Penalty} = \max(0, (\text{Mean}(Z_{\text{Humidity}}, Z_{\text{Wind}}, Z_{\text{Pressure}}, Z_{\text{Rain}}) - 1.5) \times 15)$$
    """)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Weather Analytics & ML Prediction<br>
    Developed by <b>Abakaushik</b><br>
    Artificial Intelligence Internship Project 2026
</div>
""", unsafe_allow_html=True)
