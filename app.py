import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Append project root to sys path for modular imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.evaluator import evaluate_classification_performance, get_confusion_matrix_and_report

# Page Configuration
st.set_page_config(
    page_title="Mobile Price Range Intelligence System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme & Cards)
st.markdown("""
    <style>
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #546E7A;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        color: white;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: bold;
    }
    .price-tag-0 { background-color: #4CAF50; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; }
    .price-tag-1 { background-color: #2196F3; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; }
    .price-tag-2 { background-color: #FF9800; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; }
    .price-tag-3 { background-color: #E91E63; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📱 Mobile Price Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">BITS Pilani Machine Learning Assignment 2 — End-to-End ML Pipeline Dashboard</div>', unsafe_allow_html=True)

# Helper function to safely get probabilities across sklearn versions
def safe_predict_proba(model, X):
    if not hasattr(model, 'multi_class'):
        setattr(model, 'multi_class', 'auto')
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
    except Exception:
        pass
    return None

# Cache Model and Scaler Loader
@st.cache_resource
def load_all_artifacts():
    scaler = joblib.load('model/mobile_price_scaler.joblib')
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression_model.joblib'),
        'Decision Tree': joblib.load('model/decision_tree_model.joblib'),
        'kNN': joblib.load('model/knn_classifier_model.joblib'),
        'Naive Bayes': joblib.load('model/naive_bayes_model.joblib'),
        'Random Forest': joblib.load('model/random_forest_model.joblib')
    }
    for m_name, m_obj in models.items():
        if not hasattr(m_obj, 'multi_class'):
            setattr(m_obj, 'multi_class', 'auto')
            
    metrics_df = pd.read_csv('model/model_metrics.csv')
    return scaler, models, metrics_df

try:
    scaler, models, metrics_df = load_all_artifacts()
except Exception as e:
    st.error(f"Error loading trained models: {e}. Please ensure model training has been completed.")
    st.stop()

# Sidebar Setup
st.sidebar.title("🛠️ Configuration")
selected_model_name = st.sidebar.selectbox("Select ML Classifier", list(models.keys()))
active_model = models[selected_model_name]

# Define 4 Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Single Device Predictor",
    "📁 Batch CSV Evaluator",
    "📊 Model Performance Benchmark",
    "ℹ️ Dataset & Problem Overview"
])

PRICE_MAP = {
    0: ("0 - Low Cost", "#4CAF50"),
    1: ("1 - Medium Cost", "#2196F3"),
    2: ("2 - High Cost", "#FF9800"),
    3: ("3 - Very High Cost", "#E91E63")
}

# --- TAB 1: LIVE SINGLE DEVICE PREDICTOR ---
with tab1:
    st.subheader("⚙️ Input Mobile Hardware Specifications")
    st.caption("Adjust sliders and values to estimate the price range tier in real-time.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ram = st.number_input("RAM (MB)", min_value=256, max_value=8000, value=3000, step=128)
        battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=6000, value=1500, step=100)
        px_height = st.number_input("Pixel Resolution Height", min_value=0, max_value=2000, value=600, step=50)
        px_width = st.number_input("Pixel Resolution Width", min_value=0, max_value=2500, value=1200, step=50)
        int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=512, value=32, step=2)

    with col2:
        mobile_wt = st.number_input("Mobile Weight (g)", min_value=80, max_value=250, value=140, step=5)
        clock_speed = st.slider("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        n_cores = st.slider("Processor Cores", min_value=1, max_value=8, value=4)
        talk_time = st.slider("Talk Time (Hours)", min_value=2, max_value=20, value=10)
        m_dep = st.slider("Mobile Depth (cm)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    with col3:
        fc = st.number_input("Front Camera (MP)", min_value=0, max_value=30, value=5)
        pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=50, value=12)
        sc_h = st.slider("Screen Height (cm)", min_value=5, max_value=20, value=12)
        sc_w = st.slider("Screen Width (cm)", min_value=0, max_value=18, value=6)

    with col4:
        st.write("**Connectivity Features**")
        blue = st.checkbox("Bluetooth Available", value=True)
        dual_sim = st.checkbox("Dual SIM Support", value=True)
        four_g = st.checkbox("4G Network Enabled", value=True)
        three_g = st.checkbox("3G Network Enabled", value=True)
        touch_screen = st.checkbox("Touch Screen", value=True)
        wifi = st.checkbox("Wi-Fi Connectivity", value=True)

    input_data = {
        'battery_power': battery_power, 'blue': int(blue), 'clock_speed': clock_speed,
        'dual_sim': int(dual_sim), 'fc': fc, 'four_g': int(four_g),
        'int_memory': int_memory, 'm_dep': m_dep, 'mobile_wt': mobile_wt,
        'n_cores': n_cores, 'pc': pc, 'px_height': px_height,
        'px_width': px_width, 'ram': ram, 'sc_h': sc_h, 'sc_w': sc_w,
        'talk_time': talk_time, 'three_g': int(three_g), 'touch_screen': int(touch_screen),
        'wifi': int(wifi)
    }

    if st.button("🚀 Predict Mobile Price Tier", type="primary"):
        df_single = pd.DataFrame([input_data])
        scaled_single = scaler.transform(df_single)
        pred_class = active_model.predict(scaled_single)[0]
        
        label_str, color_code = PRICE_MAP[pred_class]
        st.markdown("---")
        st.markdown(f"### Predicted Price Category: <span style='color:{color_code}; font-size:1.8rem; font-weight:bold;'>{label_str}</span>", unsafe_allow_html=True)
        
        probs_matrix = safe_predict_proba(active_model, scaled_single)
        if probs_matrix is not None:
            probs = probs_matrix[0]
            st.write("**Prediction Probabilities across Price Tiers:**")
            prob_df = pd.DataFrame({'Price Tier': [PRICE_MAP[i][0] for i in range(4)], 'Probability': probs})
            st.bar_chart(prob_df.set_index('Price Tier'))

# --- TAB 2: BATCH CSV EVALUATOR ---
with tab2:
    st.subheader("📂 Upload CSV File for Batch Evaluation & Inference")
    uploaded_file = st.file_uploader("Upload CSV dataset (e.g. test_data.csv):", type=["csv"])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.success(f"Uploaded successfully! Dataset shape: {df_upload.shape}")
        
        has_labels = 'price_range' in df_upload.columns
        if has_labels:
            X_batch = df_upload.drop(columns=['price_range'])
            y_batch = df_upload['price_range']
        else:
            X_batch = df_upload
            y_batch = None
            
        try:
            X_batch_scaled = scaler.transform(X_batch)
            preds = active_model.predict(X_batch_scaled)
            probs = safe_predict_proba(active_model, X_batch_scaled)
            
            df_out = X_batch.copy()
            df_out['Predicted_Price_Range'] = preds
            if y_batch is not None:
                df_out['Actual_Price_Range'] = y_batch
                
            st.write("Preview of Batch Predictions:")
            st.dataframe(df_out.head(50))
            
            # CSV Download
            csv_data = df_out.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Predictions CSV",
                data=csv_data,
                file_name=f"mobile_predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            if y_batch is not None:
                st.markdown("---")
                st.subheader(f"📈 Evaluation Metrics on Uploaded Data ({selected_model_name})")
                
                metrics = evaluate_classification_performance(selected_model_name, y_batch, preds, probs)
                
                m_cols = st.columns(6)
                metric_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
                for idx, k in enumerate(metric_keys):
                    with m_cols[idx]:
                        st.markdown(f"""
                            <div class="metric-badge">
                                <div class="metric-title">{k}</div>
                                <div class="metric-val">{metrics[k]:.4f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                st.markdown("<br>", unsafe_allow_html=True)
                cm_col, rep_col = st.columns([1, 1])
                
                cm, report_dict = get_confusion_matrix_and_report(y_batch, preds)
                
                with cm_col:
                    st.subheader("🎯 Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual Target")
                    st.pyplot(fig_cm)
                    
                with rep_col:
                    st.subheader("📋 Classification Report")
                    st.dataframe(pd.DataFrame(report_dict).transpose().style.format(precision=4))
                    
        except Exception as err:
            st.error(f"Error processing uploaded dataset: {err}")

# --- TAB 3: MODEL PERFORMANCE BENCHMARK ---
with tab3:
    st.subheader("📊 Cross-Validation Performance Comparison")
    st.caption("Summary of metrics across all 5 algorithms trained on the Mobile Price Classification dataset.")
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#C8E6C9', subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']).format(precision=4))
    
    fig_bench, ax_bench = plt.subplots(figsize=(10, 5))
    df_melt = metrics_df.melt(id_vars='ML Model Name', var_name='Metric', value_name='Score')
    sns.barplot(data=df_melt, x='ML Model Name', y='Score', hue='Metric', ax=ax_bench)
    ax_bench.set_ylim(0.4, 1.05)
    ax_bench.set_title("Model Benchmarking Across 6 Evaluation Metrics")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    st.pyplot(fig_bench)
    
    st.markdown("""
    ### 📝 Model Performance Insights
    * **Logistic Regression (Overall Winner 🏆)**: Achieved the highest accuracy (**98.00%**) and near-perfect AUC (**0.9995**). Standardizing features allowed the model to capitalize on strong linear relationships between RAM/pixel dimensions and price tier.
    * **Random Forest Classifier**: Showed high accuracy (**88.75%**) and an AUC of **0.9805**, proving effective at capturing complex non-linear decision boundaries.
    * **Decision Tree Classifier**: Attained **83.00%** accuracy, providing an interpretable rule-based framework.
    * **Gaussian Naive Bayes**: Reached **79.00%** accuracy. Performance is slightly constrained by the feature independence assumption.
    * **K-Nearest Neighbors (KNN)**: Achieved **67.50%** accuracy. Distance metrics face scaling challenges in 20-dimensional feature spaces.
    """)

# --- TAB 4: DATASET & PROBLEM OVERVIEW ---
with tab4:
    st.subheader("ℹ️ Dataset Specification & Problem Context")
    st.markdown("""
    - **Problem Type**: Multi-class Classification (4 Classes: 0: Low, 1: Medium, 2: High, 3: Very High)
    - **Total Dataset Size**: 2,000 instances
    - **Total Features**: 20 technical specifications
    - **Target Variable**: `price_range`
    - **Evaluation Metrics**: Accuracy, AUC (OvR), Precision, Recall, F1-Score, MCC.
    """)
