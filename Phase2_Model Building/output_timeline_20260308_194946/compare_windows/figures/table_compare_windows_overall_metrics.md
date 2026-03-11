# Compare Windows Overall Metrics Table

Data source: `fig_compare_windows_overall_values.csv`
Related figures:
- `fig_compare_windows_overall_auc.png`
- `fig_compare_windows_overall_f1.png`
- `fig_compare_windows_overall_precision.png`
- `fig_compare_windows_overall_recall.png`

| Time Window | Domain | Model | AUC (mean ± std) | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) |
|---|---|---|---|---|---|---|
| 7d | chicago_2025 | baseline_persistence | 0.937 ± 0.000 | 0.694 ± 0.000 | 0.691 ± 0.000 | 0.697 ± 0.000 |
| 7d | chicago_2025 | logistic_regression | 0.954 ± 0.000 | 0.756 ± 0.010 | 0.648 ± 0.015 | 0.907 ± 0.002 |
| 7d | chicago_2025 | random_forest | 0.952 ± 0.001 | 0.761 ± 0.007 | 0.734 ± 0.004 | 0.790 ± 0.013 |
| 7d | chicago_2025 | lstm | 0.927 ± 0.010 | 0.693 ± 0.023 | 0.610 ± 0.024 | 0.802 ± 0.023 |
| 7d | nibrs | baseline_persistence | 0.998 ± 0.000 | 0.942 ± 0.000 | 0.933 ± 0.000 | 0.951 ± 0.000 |
| 7d | nibrs | logistic_regression | 0.892 ± 0.040 | 0.561 ± 0.075 | 0.993 ± 0.004 | 0.394 ± 0.075 |
| 7d | nibrs | random_forest | 0.997 ± 0.000 | 0.951 ± 0.002 | 0.957 ± 0.004 | 0.944 ± 0.001 |
| 7d | nibrs | lstm | 0.923 ± 0.055 | 0.351 ± 0.095 | 0.446 ± 0.427 | 0.505 ± 0.190 |
| 3d | chicago_2025 | baseline_persistence | 0.895 ± 0.000 | 0.619 ± 0.000 | 0.618 ± 0.000 | 0.620 ± 0.000 |
| 3d | chicago_2025 | logistic_regression | 0.932 ± 0.000 | 0.704 ± 0.003 | 0.602 ± 0.011 | 0.849 ± 0.017 |
| 3d | chicago_2025 | random_forest | 0.930 ± 0.000 | 0.677 ± 0.007 | 0.699 ± 0.007 | 0.656 ± 0.011 |
| 3d | chicago_2025 | lstm | 0.895 ± 0.005 | 0.626 ± 0.015 | 0.586 ± 0.006 | 0.672 ± 0.028 |
| 3d | nibrs | baseline_persistence | 0.999 ± 0.000 | 0.907 ± 0.000 | 0.906 ± 0.000 | 0.909 ± 0.000 |
| 3d | nibrs | logistic_regression | 0.837 ± 0.051 | 0.404 ± 0.048 | 1.000 ± 0.000 | 0.254 ± 0.038 |
| 3d | nibrs | random_forest | 0.998 ± 0.001 | 0.928 ± 0.002 | 0.973 ± 0.004 | 0.887 ± 0.008 |
| 3d | nibrs | lstm | 0.729 ± 0.438 | 0.135 ± 0.122 | 0.381 ± 0.465 | 0.087 ± 0.076 |
