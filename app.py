"""
app.py
------
Streamlit Web Application for AI-Based Energy Consumption Forecasting.

Features:
- Dashboard: Key dataset summary metrics and state/region summaries.
- Data Visualizations: Historical trends, seasonality, and regional comparisons.
- Model Comparison: Live metrics table (MAE, RMSE, R²) and performance plots.
- Energy Predictor: Interactive user input form to forecast future energy consumption.

Author: AI Project Team
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# Import project modules
from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import prepare_features
from src.prediction import EnergyPredictor

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #2563EB;
    }
    .best-badge {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    """Loads preprocessed dataset for dashboard visualization."""
    df, meta = load_and_preprocess_data(verbose=False)
    return df, meta


@st.cache_resource
def get_predictor():
    """Instantiates the EnergyPredictor class."""
    return EnergyPredictor()


@st.cache_data
def get_metrics_and_eval():
    """Loads saved model metrics and test evaluation data, auto-training if missing."""
    metrics_path = os.path.join("models", "model_metrics.json")
    eval_path = os.path.join("models", "evaluation_data.joblib")
    best_model_path = os.path.join("models", "best_model.joblib")
    
    # Auto-train models if missing on cloud container
    if not os.path.exists(metrics_path) or not os.path.exists(eval_path) or not os.path.exists(best_model_path):
        from src.train_models import run_pipeline
        run_pipeline()
        
    if os.path.exists(metrics_path) and os.path.exists(eval_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        eval_df = joblib.load(eval_path)
        return metrics, eval_df
    return None, None


def main():
    # Header Banner
    st.markdown('<div class="main-header">⚡ AI-Based Energy Consumption Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Machine Learning system for historical analysis and future electricity demand prediction.</div>', unsafe_allow_html=True)

    # Load Data and Models
    try:
        metrics, eval_df = get_metrics_and_eval()
        df, meta = get_dataset()
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Error initializing application: {e}")
        st.stop()

    # Sidebar Information
    with st.sidebar:
        st.markdown("## ⚡ EnergyAI")
        st.title("Project Controls")
        st.info("""
        **Project**: Energy Forecasting  
        **Dataset**: Daily Power Consumption in India  
        **Unit**: Mega Units (MU)  
        *(1 MU = 1 Million kWh)*
        """)

        
        st.markdown("---")
        st.markdown("### 🔍 Model Information")
        if metrics:
            st.success(f"**Best Model**: {metrics['best_model']}")
            st.metric("Test R² Score", f"{metrics['best_metrics']['R2']:.4f}")
            st.metric("Test RMSE", f"{metrics['best_metrics']['RMSE']} MU")
        
        st.markdown("---")
        st.caption("College ML Project | Machine Learning System")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard & Overview",
        "📈 Data Visualizations",
        "🏆 Model Comparison & Evaluation",
        "🔮 Predict Energy Consumption"
    ])

    # ----------------------------------------------------
    # TAB 1: DASHBOARD & OVERVIEW
    # ----------------------------------------------------
    with tab1:
        st.subheader("📌 Dataset Summary & Key Metrics")
        
        # Metric Cards Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.caption(f"Spanning {meta['start_date']} to {meta['end_date']}")
        with col2:
            st.metric("Average Daily Usage", f"{df['Usage'].mean():.2f} MU")
            st.caption(f"Std Dev: ±{df['Usage'].std():.2f} MU")
        with col3:
            st.metric("Minimum Usage", f"{df['Usage'].min():.2f} MU")
            st.caption("Lowest recorded state day")
        with col4:
            st.metric("Maximum Usage", f"{df['Usage'].max():.2f} MU")
            st.caption("Peak recorded state day")

        st.markdown("---")
        
        # Two columns for exploration
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("#### 🏛️ Top 10 Energy Consuming States (Daily Average)")
            top_states = df.groupby('States')['Usage'].mean().sort_values(ascending=False).head(10).reset_index()
            fig_top = px.bar(
                top_states, x='Usage', y='States', orientation='h',
                color='Usage', color_continuous_scale='Blues',
                labels={'Usage': 'Average Daily Usage (MU)', 'States': 'State'},
                title="Top 10 States by Electricity Demand"
            )
            fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_top, use_container_width=True)

        with c2:
            st.markdown("#### 🌐 Regional Grid Power Distribution")
            region_dist = df.groupby('Regions')['Usage'].sum().reset_index()
            region_names = {
                'WR': 'Western Region (WR)',
                'NR': 'Northern Region (NR)',
                'SR': 'Southern Region (SR)',
                'ER': 'Eastern Region (ER)',
                'NER': 'North-Eastern Region (NER)'
            }
            region_dist['Region_Name'] = region_dist['Regions'].map(region_names).fillna(region_dist['Regions'])
            fig_pie = px.pie(
                region_dist, values='Usage', names='Region_Name',
                title="Total Energy Consumption by Regional Grid",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 📋 Raw Preprocessed Data Sample")
        st.dataframe(df.head(8), use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: DATA VISUALIZATIONS
    # ----------------------------------------------------
    with tab2:
        st.subheader("📈 Time Series & Seasonality Visualizations")
        
        # State Filter for Interactive Historical Time Series
        states_list = ["All India Total"] + sorted(df['States'].unique().tolist())
        selected_state = st.selectbox("Select State to View Historical Trend:", states_list)
        
        if selected_state == "All India Total":
            plot_df = df.groupby('ParsedDate')['Usage'].sum().reset_index()
            title_text = "All-India Daily Electricity Consumption Over Time"
        else:
            plot_df = df[df['States'] == selected_state].sort_values('ParsedDate')
            title_text = f"Daily Electricity Consumption in {selected_state} Over Time"

        fig_ts = px.line(
            plot_df, x='ParsedDate', y='Usage',
            title=title_text,
            labels={'ParsedDate': 'Date', 'Usage': 'Energy Consumption (MU)'},
            color_discrete_sequence=['#2563EB']
        )
        fig_ts.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### 📅 Monthly Seasonality Analysis")
            df['MonthName'] = df['ParsedDate'].dt.strftime('%b')
            df['MonthNum'] = df['ParsedDate'].dt.month
            monthly_df = df.groupby(['MonthNum', 'MonthName'])['Usage'].mean().reset_index().sort_values('MonthNum')
            
            fig_month = px.bar(
                monthly_df, x='MonthName', y='Usage',
                color='Usage', color_continuous_scale='Viridis',
                labels={'MonthName': 'Month', 'Usage': 'Average Daily Usage (MU)'},
                title="Average Daily Electricity Usage by Month (Summer vs Winter)"
            )
            fig_month.update_layout(height=350)
            st.plotly_chart(fig_month, use_container_width=True)

        with col_v2:
            st.markdown("#### 🏖️ Weekday vs. Weekend Consumption Pattern")
            df['DayName'] = df['ParsedDate'].dt.strftime('%A')
            df['DayOfWeek'] = df['ParsedDate'].dt.dayofweek
            day_df = df.groupby(['DayOfWeek', 'DayName'])['Usage'].mean().reset_index().sort_values('DayOfWeek')
            
            fig_day = px.bar(
                day_df, x='DayName', y='Usage',
                color='Usage', color_continuous_scale='Teal',
                labels={'DayName': 'Day of Week', 'Usage': 'Average Daily Usage (MU)'},
                title="Weekly Electricity Demand Cycle"
            )
            fig_day.update_layout(height=350)
            st.plotly_chart(fig_day, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: MODEL COMPARISON & EVALUATION
    # ----------------------------------------------------
    with tab3:
        st.subheader("🏆 Model Evaluation & Comparison on Unseen Test Data")
        
        if metrics:
            models_data = metrics["models"]
            best_model_name = metrics["best_model"]
            
            # Format DataFrame for comparison table
            comp_rows = []
            for m_name, m_vals in models_data.items():
                is_best = (m_name == best_model_name)
                comp_rows.append({
                    "Model Name": f"⭐ {m_name} (Best)" if is_best else m_name,
                    "Mean Absolute Error (MAE)": f"{m_vals['MAE']:.3f} MU",
                    "Root Mean Squared Error (RMSE)": f"{m_vals['RMSE']:.3f} MU",
                    "R² Score (Coefficient of Determination)": f"{m_vals['R2']:.4f}",
                    "Accuracy Ranking": "1st 🥇" if is_best else ("2nd 🥈" if "Random" in m_name else ("3rd 🥉" if "Linear" in m_name else "4th"))
                })
                
            comp_df = pd.DataFrame(comp_rows)
            st.table(comp_df)
            
            st.info(f"""
            💡 **Best Model Selected**: **{best_model_name}**  
            - **R² Score**: `{metrics['best_metrics']['R2']:.4f}` (Explains {metrics['best_metrics']['R2']*100:.2f}% of energy consumption variance).  
            - **RMSE**: `{metrics['best_metrics']['RMSE']} MU` (Lowest error penalty on test forecasting).  
            - **Why it won**: Gradient Boosting sequentially corrects errors from previous trees, learning subtle non-linear temporal trends across state load profiles.
            """)
            
            st.markdown("---")
            
            # Comparison Plots
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown("#### 📊 Evaluation Metrics Comparison")
                names = list(models_data.keys())
                rmses = [models_data[m]["RMSE"] for m in names]
                r2s = [models_data[m]["R2"] for m in names]
                
                fig_comp = go.Figure(data=[
                    go.Bar(name='RMSE (MU) - Lower is Better', x=names, y=rmses, marker_color='#E11D48'),
                    go.Bar(name='R² Score - Higher is Better', x=names, y=r2s, marker_color='#10B981')
                ])
                fig_comp.update_layout(barmode='group', height=360, title="RMSE vs R² Score Across Models")
                st.plotly_chart(fig_comp, use_container_width=True)

            with c_p2:
                st.markdown("#### 🎯 Actual vs Predicted Consumption (Test Set Sample)")
                if eval_df is not None:
                    sample_st = "Maharashtra" if "Maharashtra" in eval_df['States'].values else eval_df['States'].iloc[0]
                    test_sample = eval_df[eval_df['States'] == sample_st].sort_values('ParsedDate')
                    
                    fig_act_pred = go.Figure()
                    fig_act_pred.add_trace(go.Scatter(
                        x=test_sample['ParsedDate'], y=test_sample['Actual'],
                        mode='lines', name='Actual Consumption', line=dict(color='#2563EB', width=2)
                    ))
                    fig_act_pred.add_trace(go.Scatter(
                        x=test_sample['ParsedDate'], y=test_sample[f'Pred_{best_model_name}'],
                        mode='lines', name=f'Predicted ({best_model_name})', line=dict(color='#F59E0B', dash='dash', width=2)
                    ))
                    fig_act_pred.update_layout(
                        title=f"Actual vs Predicted for {sample_st}",
                        xaxis_title="Date", yaxis_title="Consumption (MU)",
                        height=360, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_act_pred, use_container_width=True)
        else:
            st.warning("Model metrics not found. Train models first using 'python -m src.train_models'.")

    # ----------------------------------------------------
    # TAB 4: ENERGY CONSUMPTION PREDICTION
    # ----------------------------------------------------
    with tab4:
        st.subheader("🔮 Predict Future Energy Consumption")
        st.write("Input the target date, select the state, and provide recent consumption signals to generate a forecast.")
        
        available_states = predictor.get_available_states()
        
        with st.form("prediction_form"):
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                input_state = st.selectbox("1. Select State / Union Territory:", available_states, index=available_states.index("Maharashtra") if "Maharashtra" in available_states else 0)
                state_info = predictor.get_state_info(input_state)
                st.caption(f"📍 Grid Region: **{state_info['region']}** | Historical Daily Average: **{state_info['mean_usage']:.1f} MU** (Min: {state_info['min_usage']:.1f}, Max: {state_info['max_usage']:.1f})")
                
                input_date = st.date_input(
                    "2. Select Target Forecast Date:",
                    value=date(2024, 7, 15),
                    min_value=date(2019, 1, 1),
                    max_value=date(2030, 12, 31)
                )
                
            with col_in2:
                model_options = ["Best Model", "Gradient Boosting", "Random Forest", "Decision Tree", "Linear Regression"]
                selected_model = st.selectbox("3. Select Machine Learning Model:", model_options)
                
                use_auto_baseline = st.checkbox("⚡ Auto-fill recent consumption with State Historical Baseline", value=True)
                
                if use_auto_baseline:
                    st.info(f"Using **{input_state}** historical median ({state_info['median_usage']:.1f} MU) for lag features.")
                    lag1_val = state_info['median_usage']
                    lag7_val = state_info['median_usage']
                    roll7_val = state_info['median_usage']
                else:
                    lag1_val = st.number_input("Yesterday's Energy Consumption (Lag 1) in MU:", value=float(state_info['median_usage']), min_value=0.1)
                    lag7_val = st.number_input("Same Day Last Week Consumption (Lag 7) in MU:", value=float(state_info['median_usage']), min_value=0.1)
                    roll7_val = (lag1_val + lag7_val) / 2
                    
            submit_btn = st.form_submit_button("⚡ Predict Energy Consumption", use_container_width=True)

        if submit_btn:
            with st.spinner("Calculating ML forecast..."):
                pred_res = predictor.predict(
                    state_name=input_state,
                    date_input=input_date,
                    usage_lag_1=lag1_val,
                    usage_lag_7=lag7_val,
                    usage_rolling_7=roll7_val,
                    model_choice=selected_model
                )
                
            # Display Prediction Card
            st.success(" Forecasting Completed Successfully!")
            
            p_col1, p_col2, p_col3 = st.columns([1.2, 1, 1])
            with p_col1:
                st.markdown(f"""
                <div style="background: #EFF6FF; border-radius: 12px; padding: 20px; border: 2px solid #3B82F6;">
                    <h4 style="margin: 0; color: #1E40AF;">Predicted Energy Consumption</h4>
                    <h1 style="margin: 10px 0; color: #1D4ED8; font-size: 2.8rem;">{pred_res['predicted_usage']} <span style="font-size: 1.2rem;">MU</span></h1>
                    <p style="margin: 0; color: #4B5563;">State: <b>{pred_res['state']}</b> | Date: <b>{pred_res['date']} ({pred_res['day_name']})</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            with p_col2:
                diff_pct = ((pred_res['predicted_usage'] - pred_res['state_median_historical']) / pred_res['state_median_historical']) * 100
                st.metric(
                    "Historical Median Comparison",
                    f"{pred_res['state_median_historical']} MU",
                    f"{diff_pct:+.1f}% vs baseline"
                )
                st.metric("Weekend Indicator", "Weekend" if pred_res['is_weekend'] else "Weekday")
                
            with p_col3:
                st.metric("Model Utilized", pred_res['model_used'])
                st.metric("Regional Grid", f"{pred_res['region']} Region")
                
            # Explanation breakdown
            with st.expander("🔍 View Detailed Prediction Feature Breakdown"):
                st.json(pred_res['input_features'])


if __name__ == "__main__":
    main()
