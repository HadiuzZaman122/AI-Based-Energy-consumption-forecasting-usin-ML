"""
app.py
------
Streamlit Web Application for AI-Based Energy Consumption Forecasting Using Machine Learning.

Features:
1. Executive Overview & Dataset Metrics.
2. Time-Series Trends & Seasonality Visualizations.
3. Machine Learning Model Evaluation (MAE, MSE, RMSE, R², MAPE, MedAE, Explained Variance).
4. Real-Time Energy Consumption Prediction Engine (Using Trained ML Pipeline).
5. Interactive Downloadable CSV Reports for Model Evaluation, Predictions, and Residuals.

Design:
- Clean, modern light theme with crisp white cards, blue/teal accents, and high-readability typography.

Author: AI Project Team
"""

import os
import json
from datetime import datetime, date
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import modular project components
from src.data_preprocessing import load_and_preprocess_data
from src.prediction import EnergyPredictor

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Theme CSS
st.markdown("""
<style>
    /* Global Base */
    .main {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Clean Header */
    .app-header {
        background: #FFFFFF;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        border-top: 4px solid #2563EB;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin: 0.3rem 0;
    }
    .stat-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-desc {
        font-size: 0.8rem;
        color: #94A3B8;
    }

    /* Prediction Result Card */
    .pred-card {
        background: #F0F9FF;
        border: 2px solid #0284C7;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);
    }
    .pred-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0369A1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .pred-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0369A1;
        margin: 0.4rem 0;
    }
    .pred-unit {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0284C7;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }

    /* Badges */
    .badge-best {
        background-color: #10B981;
        color: white;
        padding: 3px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and caches preprocessed dataset for visualization."""
    return load_and_preprocess_data(verbose=False)


@st.cache_resource
def get_predictor() -> EnergyPredictor:
    """Initializes and caches the EnergyPredictor inference engine."""
    return EnergyPredictor()


@st.cache_data
def get_metrics_and_eval() -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """Loads saved model metrics and test set evaluation predictions."""
    metrics_path = os.path.join("models", "model_metrics.json")
    eval_path = os.path.join("models", "evaluation_data.joblib")
    best_model_path = os.path.join("models", "best_model.joblib")

    # Auto-train models if missing on cloud container
    if not (os.path.exists(metrics_path) and os.path.exists(eval_path) and os.path.exists(best_model_path)):
        from src.train_models import run_pipeline
        run_pipeline()

    if os.path.exists(metrics_path) and os.path.exists(eval_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        eval_df = joblib.load(eval_path)
        return metrics, eval_df
    return None, None


def main():
    # -------------------------------------------------------------------------
    # 1. APPLICATION HEADER
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="app-header">
        <div class="app-title">⚡ AI-Based Energy Consumption Forecasting</div>
        <p class="app-subtitle">
            An end-to-end Machine Learning system for historical power demand analytics and time-series forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load Data and Models
    try:
        df, meta = get_dataset()
        metrics, eval_df = get_metrics_and_eval()
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # 2. SIDEBAR CONTROLS & INFORMATION
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚡ System Overview")
        st.info(f"""
        **Target**: Daily Electricity Usage  
        **Unit**: Mega Units (MU) *(1 MU = 1M kWh)*  
        **Time Span**: {meta['start_date']} to {meta['end_date']}  
        **Total Records**: {meta['final_rows']:,}  
        **Regions**: {meta['regions_count']} Grids ({', '.join(meta['regions'])})  
        **States/UTs**: {meta['states_count']}
        """)

        st.markdown("---")
        st.markdown("### 🏆 Top Model Status")
        if metrics:
            best_m = metrics.get('best_model', 'N/A')
            best_r2 = metrics['best_metrics']['R2']
            best_rmse = metrics['best_metrics']['RMSE']
            st.success(f"**Best Model**: {best_m}")
            st.metric("Test R² Score", f"{best_r2:.4f}")
            st.metric("Test RMSE", f"{best_rmse} MU")

        st.markdown("---")
        st.caption("College ML Project | Pair Programming AI Assistant")

    # -------------------------------------------------------------------------
    # 3. MAIN DASHBOARD TABS
    # -------------------------------------------------------------------------
    tab_overview, tab_viz, tab_models, tab_predict = st.tabs([
        "📊 Dashboard & Overview",
        "📈 Time Series & Seasonality",
        "🏆 Model Comparison & Evaluation",
        "🔮 Predict Energy Consumption"
    ])

    # =========================================================================
    # TAB 1: OVERVIEW & METRICS
    # =========================================================================
    with tab_overview:
        st.subheader("📌 Key Dataset Metrics & Summary")

        # Top Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Records</div>
                <div class="stat-value">{meta['final_rows']:,}</div>
                <div class="stat-desc">{meta['total_days']} Days ({meta['start_date']} to {meta['end_date']})</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Average Daily Usage</div>
                <div class="stat-value">{meta['mean_usage']:.2f} <span style="font-size:1rem;">MU</span></div>
                <div class="stat-desc">Std Dev: ±{meta['std_usage']:.2f} MU</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Minimum Daily Usage</div>
                <div class="stat-value">{meta['min_usage']:.2f} <span style="font-size:1rem;">MU</span></div>
                <div class="stat-desc">Lowest single-state record</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Maximum Daily Usage</div>
                <div class="stat-value">{meta['max_usage']:.2f} <span style="font-size:1rem;">MU</span></div>
                <div class="stat-desc">Peak single-state demand</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Visualizations
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### 🏛️ Top 10 Energy Consuming States (Daily Average)")
            top10 = df.groupby('States')['Usage'].mean().sort_values(ascending=False).head(10).reset_index()
            fig_top10 = px.bar(
                top10, x='Usage', y='States', orientation='h',
                color='Usage', color_continuous_scale='Blues',
                labels={'Usage': 'Average Daily Usage (MU)', 'States': 'State'},
                title="Top 10 States by Electricity Demand"
            )
            fig_top10.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=360,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF'
            )
            st.plotly_chart(fig_top10, use_container_width=True)

        with col_g2:
            st.markdown("#### 🌐 Regional Grid Power Distribution")
            reg_df = df.groupby('Regions')['Usage'].sum().reset_index()
            region_map = {
                'WR': 'Western Region (WR)',
                'NR': 'Northern Region (NR)',
                'SR': 'Southern Region (SR)',
                'ER': 'Eastern Region (ER)',
                'NER': 'North-Eastern Region (NER)'
            }
            reg_df['Region_Name'] = reg_df['Regions'].map(region_map).fillna(reg_df['Regions'])
            fig_pie = px.pie(
                reg_df, values='Usage', names='Region_Name',
                title="Total Energy Consumption by Regional Grid",
                color_discrete_sequence=px.colors.qualitative.Safe,
                hole=0.4
            )
            fig_pie.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='#FFFFFF'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 📋 Clean Preprocessed Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # =========================================================================
    # TAB 2: TIME SERIES & SEASONALITY
    # =========================================================================
    with tab_viz:
        st.subheader("📈 Time Series Trends & Seasonality Patterns")

        # State selector for historical trend
        state_options = ["All India Total"] + sorted(df['States'].unique().tolist())
        sel_state = st.selectbox("Select State to View Historical Time Series:", state_options)

        if sel_state == "All India Total":
            trend_df = df.groupby('ParsedDate')['Usage'].sum().reset_index()
            chart_title = "All-India Total Daily Electricity Consumption Over Time"
        else:
            trend_df = df[df['States'] == sel_state].sort_values('ParsedDate')
            chart_title = f"Daily Electricity Consumption in {sel_state} Over Time"

        fig_ts = px.line(
            trend_df, x='ParsedDate', y='Usage',
            title=chart_title,
            labels={'ParsedDate': 'Date', 'Usage': 'Energy Consumption (MU)'},
            color_discrete_sequence=['#2563EB']
        )
        fig_ts.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF'
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")

        # Seasonality Breakdown
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("#### 📅 Monthly Seasonality")
            df_season = df.copy()
            df_season['MonthName'] = df_season['ParsedDate'].dt.strftime('%b')
            df_season['MonthNum'] = df_season['ParsedDate'].dt.month
            monthly = df_season.groupby(['MonthNum', 'MonthName'])['Usage'].mean().reset_index().sort_values('MonthNum')

            fig_month = px.bar(
                monthly, x='MonthName', y='Usage',
                color='Usage', color_continuous_scale='Teal',
                labels={'MonthName': 'Month', 'Usage': 'Avg Usage (MU)'},
                title="Average Daily Consumption by Month"
            )
            fig_month.update_layout(height=340, paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
            st.plotly_chart(fig_month, use_container_width=True)

        with v_col2:
            st.markdown("#### 🏖️ Day-of-Week Demand Cycle")
            df_season['DayName'] = df_season['ParsedDate'].dt.strftime('%A')
            df_season['DayOfWeek'] = df_season['ParsedDate'].dt.dayofweek
            daily = df_season.groupby(['DayOfWeek', 'DayName'])['Usage'].mean().reset_index().sort_values('DayOfWeek')

            fig_daily = px.bar(
                daily, x='DayName', y='Usage',
                color='Usage', color_continuous_scale='Blues',
                labels={'DayName': 'Day of Week', 'Usage': 'Avg Usage (MU)'},
                title="Average Daily Consumption Across Weekdays"
            )
            fig_daily.update_layout(height=340, paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
            st.plotly_chart(fig_daily, use_container_width=True)

    # =========================================================================
    # TAB 3: MODEL COMPARISON & EVALUATION
    # =========================================================================
    with tab_models:
        st.subheader("🏆 Machine Learning Model Evaluation on Unseen Test Data")

        if metrics and eval_df is not None:
            models_data = metrics["models"]
            best_model_name = metrics["best_model"]

            # Format Comparison Table
            comp_list = []
            for m_name, m_stats in models_data.items():
                is_best = (m_name == best_model_name)
                comp_list.append({
                    "Model Name": f"⭐ {m_name} (Best)" if is_best else m_name,
                    "MAE (MU)": m_stats.get("MAE", 0.0),
                    "MSE (MU²)": m_stats.get("MSE", 0.0),
                    "RMSE (MU)": m_stats.get("RMSE", 0.0),
                    "R² Score": m_stats.get("R2", 0.0),
                    "MAPE (%)": f"{m_stats.get('MAPE', 0.0):.2f}%",
                    "MedAE (MU)": m_stats.get("MedAE", 0.0),
                    "Explained Variance": m_stats.get("Explained_Variance", 0.0),
                    "Status": "Best Model 🥇" if is_best else "Trained"
                })

            comp_df = pd.DataFrame(comp_list)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Best Model Rationale Card
            best_info = metrics['best_metrics']
            st.info(f"""
            💡 **Best Performing Model**: **{best_model_name}**  
            - **R² Score**: `{best_info['R2']:.4f}` (Explains {best_info['R2']*100:.2f}% of energy consumption variance on unseen future data).  
            - **RMSE**: `{best_info['RMSE']} MU` (Lowest error penalty on peak load forecasting).  
            - **MAE**: `{best_info['MAE']} MU` | **MAPE**: `{best_info.get('MAPE', 0.0)}%`.  
            - **Rationale**: Tree-ensemble boosting effectively captures nonlinear regional load inertia and seasonal temperature correlations.
            """)

            st.markdown("---")

            # Comparative Visualizations
            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                st.markdown("#### 📊 Evaluation Error & R² Comparison")
                names = list(models_data.keys())
                rmses = [models_data[m]["RMSE"] for m in names]
                r2s = [models_data[m]["R2"] for m in names]

                fig_mcomp = go.Figure(data=[
                    go.Bar(name='RMSE (MU) - Lower is Better', x=names, y=rmses, marker_color='#DC2626'),
                    go.Bar(name='R² Score - Higher is Better', x=names, y=r2s, marker_color='#10B981')
                ])
                fig_mcomp.update_layout(
                    barmode='group', height=360,
                    title="Model Accuracy: RMSE vs R²",
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF'
                )
                st.plotly_chart(fig_mcomp, use_container_width=True)

            with mc_col2:
                st.markdown("#### 🎯 Actual vs Predicted Time Series (Test Set)")
                sample_state = "Maharashtra" if "Maharashtra" in eval_df['States'].values else eval_df['States'].iloc[0]
                test_sample = eval_df[eval_df['States'] == sample_state].sort_values('ParsedDate')

                fig_act = go.Figure()
                fig_act.add_trace(go.Scatter(
                    x=test_sample['ParsedDate'], y=test_sample['Actual'],
                    mode='lines', name='Actual Consumption', line=dict(color='#2563EB', width=2.2)
                ))
                fig_act.add_trace(go.Scatter(
                    x=test_sample['ParsedDate'], y=test_sample[f'Pred_{best_model_name}'],
                    mode='lines', name=f'Predicted ({best_model_name})', line=dict(color='#EA580C', dash='dash', width=2.0)
                ))
                fig_act.update_layout(
                    title=f"Actual vs Predicted in {sample_state}",
                    xaxis_title="Date", yaxis_title="Consumption (MU)",
                    height=360, margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF'
                )
                st.plotly_chart(fig_act, use_container_width=True)

            st.markdown("---")

            # Residual Error Distribution
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("#### 📉 Residual Error Distribution (Actual - Predicted)")
                residuals = eval_df['Actual'] - eval_df['Predicted']
                fig_res = px.histogram(
                    residuals, nbins=40,
                    title=f"Error Distribution ({best_model_name})",
                    labels={'value': 'Residual Error (MU)'},
                    color_discrete_sequence=['#0D9488']
                )
                fig_res.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
                fig_res.update_layout(height=340, paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
                st.plotly_chart(fig_res, use_container_width=True)

            with res_col2:
                st.markdown("#### 🔍 Residuals vs Predicted Scatter Plot")
                fig_scat = px.scatter(
                    eval_df, x='Predicted', y='Residual',
                    color='States',
                    title="Residuals vs Predicted Values",
                    labels={'Predicted': 'Predicted (MU)', 'Residual': 'Residual (Actual - Predicted)'}
                )
                fig_scat.add_hline(y=0, line_dash="dash", line_color="red")
                fig_scat.update_layout(height=340, showlegend=False, paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
                st.plotly_chart(fig_scat, use_container_width=True)

            # Download Section
            st.markdown("---")
            st.markdown("### 📥 Downloadable Evaluation Datasets")
            d_col1, d_col2, d_col3 = st.columns(3)

            with d_col1:
                comp_csv = comp_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Model Metrics CSV",
                    data=comp_csv,
                    file_name="model_evaluation_metrics.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with d_col2:
                act_cols = [c for c in ['ParsedDate', 'States', 'Regions', 'Actual', 'Predicted', 'Absolute_Error', 'Percentage_Error'] if c in eval_df.columns]
                act_csv = eval_df[act_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Actual vs Predicted CSV",
                    data=act_csv,
                    file_name="actual_vs_predicted_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with d_col3:
                res_cols = [c for c in ['ParsedDate', 'States', 'Actual', 'Predicted', 'Residual', 'Absolute_Error'] if c in eval_df.columns]
                res_csv = eval_df[res_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Residual Analysis CSV",
                    data=res_csv,
                    file_name="residual_error_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning("Model evaluation data not available. Please run training pipeline.")

    # =========================================================================
    # TAB 4: REAL PREDICTION ENGINE
    # =========================================================================
    with tab_predict:
        st.subheader("🔮 Predict Future Energy Consumption")
        st.write("Enter the target state, date, and consumption context to generate real machine learning forecasts.")

        available_states = predictor.get_available_states()
        default_state_idx = available_states.index("Maharashtra") if "Maharashtra" in available_states else 0

        # Prediction Form Card
        with st.form("forecasting_form"):
            in_col1, in_col2 = st.columns(2)

            with in_col1:
                form_state = st.selectbox(
                    "1. Target State / Union Territory:",
                    available_states,
                    index=default_state_idx
                )
                st_profile = predictor.get_state_info(form_state)
                st.caption(
                    f"📍 Region: **{st_profile['region']}** | "
                    f"Historical Median: **{st_profile['median_usage']:.1f} MU** "
                    f"(Min: {st_profile['min_usage']:.1f} MU, Max: {st_profile['max_usage']:.1f} MU)"
                )

                form_date = st.date_input(
                    "2. Target Forecast Date:",
                    value=date(2024, 7, 15),
                    min_value=date(2019, 1, 1),
                    max_value=date(2030, 12, 31)
                )

            with in_col2:
                model_choices = ["Best Model", "XGBoost", "Gradient Boosting", "Random Forest", "Decision Tree", "Linear Regression"]
                form_model = st.selectbox("3. Select Machine Learning Model:", model_choices)

                auto_baseline = st.checkbox(
                    "⚡ Auto-fill recent consumption context with State Historical Baseline",
                    value=True
                )

                if auto_baseline:
                    st.info(f"Using **{form_state}** historical baseline median ({st_profile['median_usage']:.1f} MU).")
                    form_lag1 = st_profile['median_usage']
                    form_lag7 = st_profile['median_usage']
                    form_roll7 = st_profile['median_usage']
                else:
                    form_lag1 = st.number_input(
                        "Yesterday's Usage (Lag 1) in MU:",
                        value=float(st_profile['median_usage']),
                        min_value=0.1
                    )
                    form_lag7 = st.number_input(
                        "Same Day Last Week Usage (Lag 7) in MU:",
                        value=float(st_profile['median_usage']),
                        min_value=0.1
                    )
                    form_roll7 = (form_lag1 + form_lag7) / 2

            predict_submit = st.form_submit_button("⚡ Predict Energy Consumption", use_container_width=True)

        if predict_submit:
            with st.spinner("Calculating ML forecast using trained model..."):
                pred_output = predictor.predict(
                    state_name=form_state,
                    date_input=form_date,
                    usage_lag_1=form_lag1,
                    usage_lag_7=form_lag7,
                    usage_rolling_7=form_roll7,
                    model_choice=form_model
                )

            st.success(" Forecasting Completed Successfully!")

            # Prediction Output Display
            out_c1, out_c2, out_c3 = st.columns([1.3, 1, 1])

            with out_c1:
                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-title">Forecasted Electricity Demand</div>
                    <div class="pred-value">
                        {pred_output['predicted_usage']:.2f} <span class="pred-unit">MU</span>
                    </div>
                    <div style="color: #475569; font-size: 0.92rem; margin-top: 5px;">
                        <b>{pred_output['state']}</b> | <b>{pred_output['date']} ({pred_output['day_name']})</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with out_c2:
                diff = pred_output['diff_vs_baseline_pct']
                st.metric(
                    "Comparison to State Baseline",
                    f"{pred_output['state_median_historical']} MU",
                    f"{diff:+.1f}% vs baseline",
                    delta_color="inverse" if abs(diff) > 20 else "normal"
                )
                st.metric(
                    "Calendar Context",
                    "🏖️ Weekend" if pred_output['is_weekend'] else "🏢 Weekday"
                )

            with out_c3:
                st.metric("Model Utilized", pred_output['model_used'])
                st.metric("Regional Grid", f"{pred_output['region']} Region")

            with st.expander("🔍 Inspect Full Engineered Input Feature Vector"):
                st.json(pred_output['input_features'])


if __name__ == "__main__":
    main()
