# Academic Feature Engineering Specifications

## Feature Vectors
1. **Attendance Rate:** Normalized to `[0.0, 1.0]`.
2. **Study Hours per Week:** Bounded continuous variable `[0, 50]`.
3. **Prior Test Percentile:** Min-max scaled historical average.

Optimized for linear ridge regression and gradient boosted decision trees.
