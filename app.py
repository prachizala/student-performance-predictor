"""Student Performance Predictor & Academic Risk Dashboard - Streamlit Web UI.

Built with Python, Scikit-Learn logic, and Streamlit.
Allows educators, academic advisors, and students to predict final semester grades,
identify early academic risks, and simulate 'what-if' study interventions.
"""

import os
import csv
import streamlit as st
import pandas as pd

from model_engine import (
    predict_final_score,
    get_feature_importances,
    load_dataset,
    compute_metrics,
    benchmark_models,
    DATA_PATH
)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Student Academic Performance & Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Predictive Analytics & Early Warning Decision Suite | Built by Prachi Zala</div>', unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/student-registration.png", width=120)
    st.subheader("📝 Student Profile Parameters")

    study_hours = st.slider("Weekly Study Hours (Excl. Lectures):", min_value=1.0, max_value=40.0, value=18.0, step=0.5)
    attendance_pct = st.slider("Lecture Attendance Rate (%):", min_value=40.0, max_value=100.0, value=85.0, step=1.0)
    past_score = st.slider("Previous Semester Score (%):", min_value=30.0, max_value=100.0, value=74.0, step=1.0)
    sleep_hours = st.slider("Average Nightly Sleep (Hours):", min_value=4.0, max_value=10.0, value=7.5, step=0.5)
    assignments_pct = st.slider("Assignment Submission Rate (%):", min_value=30.0, max_value=100.0, value=88.0, step=1.0)
    extracurricular = st.checkbox("Active in Extracurriculars / Sports", value=True)

    st.markdown("---")
    st.markdown("### 👩‍💻 Developer")
    st.markdown("**Prachi Zala**  \nBCA Student, Ahmedabad  \n[GitHub Profile](https://github.com/prachizala)")

# Calculate Prediction
pred = predict_final_score(
    study_hours=study_hours,
    attendance_pct=attendance_pct,
    past_score=past_score,
    sleep_hours=sleep_hours,
    assignments_pct=assignments_pct,
    extracurricular=1 if extracurricular else 0
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Prediction & Advisory",
    "🔮 What-If Scenario Simulator",
    "📈 Cohort Analytics & Dataset",
    "⚖️ Model Comparison Benchmark"
])

# -----------------------------------------------------------------------------
# TAB 1: PREDICTION & RISK ADVISORY
# -----------------------------------------------------------------------------
with tab1:
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

    with col_metric1:
        st.metric("Predicted Final Grade", f"{pred['predicted_score']:.1f}%", delta=f"{pred['predicted_score'] - past_score:+.1f}% vs Past")
    with col_metric2:
        st.metric("Letter Grade", pred["grade"])
    with col_metric3:
        st.metric("Risk Assessment", pred["risk_level"])
    with col_metric4:
        st.metric("Academic Standing", pred["status"].split("(")[0])

    if pred["predicted_score"] >= 75.0:
        st.success(f"✅ **Positive Forecast:** Student is in good standing ({pred['status']}).")
    elif pred["predicted_score"] >= 55.0:
        st.warning(f"⚠️ **Moderate Risk:** Academic advisory recommended. Focus on study hours and assignment discipline.")
    else:
        st.error(f"🚨 **High Risk Alert:** Student is at risk of failing the semester. Immediate faculty intervention required.")

    st.markdown("---")
    col_factors, col_importances = st.columns(2)

    with col_factors:
        st.subheader("Score Composition Breakdown")
        factors_df = pd.DataFrame(list(pred["factors"].items()), columns=["Factor", "Points Contributed"])
        st.dataframe(factors_df, use_container_width=True)

    with col_importances:
        st.subheader("Global Feature Importance (OLS/ML)")
        imp_data = get_feature_importances()
        imp_df = pd.DataFrame(imp_data)
        st.bar_chart(imp_df.set_index("feature"))

# -----------------------------------------------------------------------------
# TAB 2: WHAT-IF SCENARIO SIMULATOR
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Interactive Intervention Simulator")
    st.markdown("Simulate how improving study discipline or attendance changes the predicted outcome.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("#### Baseline Scenario (Current)")
        st.write(f"- Study Hours: **{study_hours} hrs/wk**")
        st.write(f"- Attendance: **{attendance_pct}%**")
        st.write(f"- Predicted: **{pred['predicted_score']:.1f}% ({pred['grade']})**")

    with col_sim2:
        st.markdown("#### Target Scenario (Intervention)")
        add_study = st.slider("Additional Weekly Study Hours:", 0.0, 15.0, 4.0, 0.5)
        add_attend = st.slider("Attendance Improvement (% points):", 0.0, 30.0, 8.0, 1.0)

        target_pred = predict_final_score(
            study_hours=min(40.0, study_hours + add_study),
            attendance_pct=min(100.0, attendance_pct + add_attend),
            past_score=past_score,
            sleep_hours=sleep_hours,
            assignments_pct=assignments_pct,
            extracurricular=1 if extracurricular else 0
        )

        st.metric(
            label="Projected New Grade",
            value=f"{target_pred['predicted_score']:.1f}% ({target_pred['grade']})",
            delta=f"{target_pred['predicted_score'] - pred['predicted_score']:+.1f}% Growth",
            delta_color="normal"
        )

# -----------------------------------------------------------------------------
# TAB 3: COHORT ANALYTICS & DATASET
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Dataset & Historical Performance Cohort")
    if os.path.exists(DATA_PATH):
        df_cohort = pd.read_csv(DATA_PATH)
        st.write(f"Showing **{len(df_cohort)}** curated student training records:")
        st.dataframe(df_cohort.head(15), use_container_width=True)

        st.markdown("#### Summary Statistics")
        st.dataframe(df_cohort.describe().round(2), use_container_width=True)
    else:
        st.info("Dataset file not found.")

# -----------------------------------------------------------------------------
# TAB 4: MODEL COMPARISON BENCHMARK
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Model Comparison Benchmark (OLS vs Random Forest)")
    st.markdown(
        "Evaluate regression predictive accuracy and statistical goodness-of-fit across "
        "Ordinary Least Squares (OLS) Linear Regression, Random Forest Regressor, and Decision Tree baselines."
    )

    benchmarks = benchmark_models()

    # Metric comparison cards for the top two models
    ols_bench = next((b for b in benchmarks if "OLS" in b["model"]), None)
    rf_bench = next((b for b in benchmarks if "Random Forest" in b["model"]), None)

    if ols_bench and rf_bench:
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            st.metric("OLS R² Score", f"{ols_bench['r2']:.4f}")
        with bcol2:
            st.metric("Random Forest R² Score", f"{rf_bench['r2']:.4f}")
        with bcol3:
            st.metric("OLS MAE", f"{ols_bench['mae']:.2f} pts")
        with bcol4:
            st.metric("Random Forest MAE", f"{rf_bench['mae']:.2f} pts")

    st.markdown("#### Comprehensive Benchmark Comparison Table")
    bench_table_data = [
        {
            "Model Architecture": b["model"],
            "R² Score (Variance Explained)": f"{b['r2']:.4f}",
            "Mean Absolute Error (MAE)": f"{b['mae']:.2f}",
            "Root Mean Squared Error (RMSE)": f"{b['rmse']:.2f}",
            "Architecture & Behavioral Notes": b["description"]
        }
        for b in benchmarks
    ]
    st.dataframe(pd.DataFrame(bench_table_data), use_container_width=True)

    st.info(
        "💡 **Engineering Trade-off Insight:** While Random Forest and ensemble methods capture complex non-linear interactions, "
        "OLS Linear Regression provides exact mathematical feature weights, transparent coefficient auditing for academic advisors, "
        "and sub-millisecond deterministic inference."
    )
