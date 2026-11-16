# Student Academic Performance & Risk Predictor 🎓📊

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-6%20Passing-10B981?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](LICENSE)

> **Predictive machine learning regression engine and early academic risk advisory dashboard evaluating study habits, attendance, and assignment discipline.**

---

## 🎯 The Problem It Solves

In undergraduate programs, academic instructors and university advisors frequently discover student learning difficulties only after mid-term exams or final semester results are published—when it is often too late for meaningful academic recovery.

**Student Performance Predictor** bridges this gap by providing an early warning decision-support system:
1. **Multi-Factor Academic Evaluation:** Analyzes key leading indicators (weekly self-study hours, lecture attendance, historical baselines, assignment completion rates, and sleep duration).
2. **Early Academic Intervention:** Categorizes student trajectories into clear risk tiers (*Low Risk*, *Moderate*, *Elevated Risk*, and *Critical Intervention*).
3. **What-If Scenario Simulation:** Enables advisors and students to model exact behavioral changes (e.g., *"How much will an extra 4 hours of weekly study boost my final semester score?"*).

---

## 📸 Architecture & Dashboard Experience

```
+-----------------------------------------------------------------------------------------+
|                               SYSTEM ARCHITECTURE & PIPELINE                            |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|   [ Student Profile Inputs ]                                                            |
|   * Study Hours / Week                                                                  |
|   * Lecture Attendance %        ────────>  [ Feature Preprocessing ]                    |
|   * Past Exam Baseline %                   - Imputation & Bounds Checking               |
|   * Sleep Duration                         - Normalized Weight Multipliers              |
|   * Assignment Submission %                                                             |
|   * Extracurriculars                                 |                                  |
|                                                      v                                  |
|                                            [ Predictive ML Engine ]                     |
|                                            - Ordinary Least Squares (OLS)               |
|                                            - Multiple Linear Regression                 |
|                                            - Metric Evaluation (R², RMSE, MAE)          |
|                                                      |                                  |
|                                                      v                                  |
|   +---------------------------------------------------------------------------------+   |
|   |                       INTERACTIVE STREAMLIT DASHBOARD                           |   |
|   |   * Real-Time Projected Score (0-100%) & Letter Grade Allocation (A+, A, B, C, F)|   |
|   |   * Academic Risk Classification Badge                                          |   |
|   |   * Factor Contribution Waterfall Breakdown                                     |   |
|   |   * Interactive What-If Intervention Simulator                                  |   |
|   +---------------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------+
```

```
+------------------------------------+   +------------------------------------+
|       PREDICTION & ADVISORY        |   |          WHAT-IF SIMULATOR         |
+------------------------------------+   +------------------------------------+
| Predicted Final Grade: 84.6% (A)   |   | Baseline: 84.6%                    |
| Standing: Very Good (On Track)     |   | Target Intervention:               |
| Delta: +10.6% vs Past Semester     |   | +4.0 Study Hours/Week              |
| Risk Level: Low Risk               |   | +8.0% Attendance Improvement       |
|                                    |   |                                    |
| Factor Contribution:               |   | Projected Grade: 91.2% (A+)        |
| * Past Exam Baseline: 23.7 pts     |   | Net Expected Growth: +6.6% 🔥      |
| * Attendance Rate:    23.8 pts     |   |                                    |
| * Study Hours:        13.5 pts     |   | [ 📥 Download Advisory Summary ]   |
+------------------------------------+   +------------------------------------+
```

---

## 🧮 Mathematical Formulation

### 1. Multiple Linear Regression Model
The predicted final exam score $\hat{y}$ is formulated as:

$$\hat{y} = w_0 + \sum_{i=1}^{p} w_i x_i + \epsilon$$

Where $x_i$ represents candidate features (study hours, attendance percentage, baseline scores) and $w_i$ denotes the learned regression coefficients.

### 2. Model Evaluation Metrics
Model goodness-of-fit and error dispersion are evaluated using:

- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

---

## ✨ Key Features

* 📈 **Continuous Grade Forecast:** Predicts real-time grade trajectories on a continuous scale ($0.0\%$ to $100.0\%$) with corresponding letter grades ($A+, A, B, C, F$).
* 🚨 **Early Academic Risk Tiers:** Automated threshold alarms highlighting students needing immediate tutoring or faculty mentoring.
* ⚖️ **Model Comparison Benchmark:** Comparative evaluation matrix benchmarking Ordinary Least Squares (OLS) Linear Regression against Decision Trees and Random Forest ensembles across $R^2$, MAE, and RMSE metrics.
* 🔮 **Interactive What-If Scenario Planner:** Dynamic sliders allowing students to test actionable improvements before midterm exams.
* 📊 **Cohort Analytics & Distributions:** Summary metrics and distribution visualizations across 500 curated student records.
* 🧪 **Automated Test Coverage:** Unit test suite covering mathematical bounds, feature weights, monotonic study properties, and CSV schema validation.

---

## 📂 Project Structure

```
student-performance-predictor/
│
├── app.py                      # Main Streamlit web dashboard
├── model_engine.py             # Core mathematical modeling and prediction logic
├── generate_data.py            # Academic dataset generator (500 records)
├── requirements.txt            # Python dependencies (Streamlit, Pandas, Scikit-Learn)
├── .gitignore                  # Clean repository ignore rules
├── README.md                   # Complete architectural guide and formulas
│
├── data/
│   └── student_performance_data.csv   # Curated training and evaluation dataset
│
└── tests/
    └── test_predictor.py       # Automated unit tests (6/6 passing)
```

---

## 🚀 Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/prachizala/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Set up Python Environment
```bash
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m unittest tests/test_predictor.py
```
*(All 6 tests will pass in under 0.20 seconds).*

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions

### Deploy to Streamlit Community Cloud (Free)
1. Push this repository to your GitHub account (`github.com/prachizala/student-performance-predictor`).
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy!** Your interactive web dashboard is live with a public URL in 60 seconds.

---

## 👤 Author & Maintainer

**Prachi Zala (Zala Prachi)**  
*Full Stack Developer at Udaan*  
- **GitHub:** [@prachizala](https://github.com/prachizala)  
- **LinkedIn:** [linkedin.com/in/zala-prachi-74a08b334](https://www.linkedin.com/in/zala-prachi-74a08b334/)  
- **Portfolio:** [prachizala.github.io/prachi-portfolio](https://prachizala.github.io/prachi-portfolio/)  
- **Location:** Ahmedabad, Gujarat, India
