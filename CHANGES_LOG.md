# Changes Log - Work Package 4: Small Feature Upgrades

## Summary of Changes

### 1. Model Engine Benchmark Function (`model_engine.py`)
- Implemented `benchmark_models(csv_path: str = DATA_PATH) -> List[Dict[str, Any]]`:
  - Evaluates goodness-of-fit and error dispersion ($R^2$, MAE, RMSE) across three models on the student dataset:
    1. **Ordinary Least Squares (OLS) Linear Regression** (baseline parametric model).
    2. **Random Forest Regressor** (bagged non-linear ensemble).
    3. **Decision Tree Regressor** (recursive partitioning benchmark).
  - Supports scikit-learn when available and provides an analytical pure-Python fallback for environments with zero third-party dependencies.

### 2. Model Comparison Benchmark Dashboard Tab (`app.py`)
- Added **Tab 4: ⚖️ Model Comparison Benchmark** to the Streamlit web dashboard.
- Surfaces side-by-side metric comparison cards (OLS vs. Random Forest $R^2$ and MAE).
- Renders an interactive comparison table detailing each algorithm's performance metrics and architectural trade-offs.
- Includes an engineering callout explaining why OLS is maintained for production transparency and instant inference while Random Forest benchmarks non-linear upper bounds.

### 3. Automated Unit Test Suite (`tests/test_predictor.py`)
- Added `test_benchmark_models_comparison` verifying:
  - Return structure of `benchmark_models()`.
  - Presence of OLS, Random Forest, and Decision Tree.
  - Validity of $R^2 > 0.5$, MAE $> 0$, and RMSE $> 0$.
- All 6 unit tests pass 100% in ~0.15s (`Ran 6 tests ... OK`).

### 4. Documentation (`README.md`)
- Updated test badge and project structure to 6/6 tests passing.
- Documented the Model Comparison Benchmark feature in Key Features.

## Items Skipped & Rationale
- None. All requested features and benchmarks were implemented and verified.

## Manual Steps Needed
- None. To view the new benchmark tab, run `streamlit run app.py` and navigate to Tab 4.
