# IT5006 Phase 2 Report (Time Window Comparison)

## Experiment Setup
- Two time windows were evaluated with the same preprocessing/mapping/model family:
  - `7d`: week-start aggregation (original setting).
  - `3d`: anchor-based fixed bins with shared anchor.
- Shared anchor date for 3-day bins: `2015-01-01`.
- Region strategy: `agency`.

## Thresholds By Time Window
| time_window | threshold_type | threshold |
| --- | --- | --- |
| 3d | global | 76 |
| 7d | global | 177 |

## Metrics Comparison (Chicago2025 and NIBRS)
| time_window | dataset | model | precision | recall | f1 | auc_roc |
| --- | --- | --- | --- | --- | --- | --- |
| 3d | chicago_2025 | baseline_persistence | 0.6175496688741722 | 0.6196013289036545 | 0.6185737976782753 | 0.8949656918897455 |
| 3d | chicago_2025 | logistic_regression | 0.601813689620202 | 0.849390919158361 | 0.7043343152944052 | 0.9322119637512866 |
| 3d | chicago_2025 | lstm | 0.5859253005777804 | 0.6716500553709857 | 0.6257224499285613 | 0.895126496278289 |
| 3d | chicago_2025 | random_forest | 0.698977822563745 | 0.655592469545958 | 0.6765502987964632 | 0.9297412180093297 |
| 3d | nibrs | baseline_persistence | 0.9058413251961639 | 0.9086139046786182 | 0.9072254966164593 | 0.9987551692918574 |
| 3d | nibrs | logistic_regression | 1.0 | 0.2540445999125492 | 0.40419636059737035 | 0.836688464694752 |
| 3d | nibrs | lstm | 0.3814687320711417 | 0.08730505757178253 | 0.13453108863065946 | 0.7289976845107757 |
| 3d | nibrs | random_forest | 0.9725444105456035 | 0.8867512024486226 | 0.9276421471835364 | 0.997714804794632 |
| 7d | chicago_2025 | baseline_persistence | 0.6910569105691057 | 0.6967213114754097 | 0.6938775510204082 | 0.9369264744301483 |
| 7d | chicago_2025 | logistic_regression | 0.6480687545227605 | 0.907103825136612 | 0.7559309078148146 | 0.9543560729758562 |
| 7d | chicago_2025 | lstm | 0.6104413985586206 | 0.8019125683060109 | 0.6931388603566164 | 0.926907805934731 |
| 7d | chicago_2025 | random_forest | 0.7344233249564535 | 0.7896174863387978 | 0.7609945205436112 | 0.9521395293144362 |
| 7d | nibrs | baseline_persistence | 0.9332603938730853 | 0.9509476031215162 | 0.9420209828823854 | 0.9982853363943279 |
| 7d | nibrs | logistic_regression | 0.993055602530483 | 0.39390561129691565 | 0.5611965915719974 | 0.8924460164418395 |
| 7d | nibrs | lstm | 0.44607102013443906 | 0.5046451133407656 | 0.35084274746294947 | 0.922603394778894 |
| 7d | nibrs | random_forest | 0.957439244597611 | 0.9442586399108138 | 0.9507983187619168 | 0.9968248518466455 |

## Generalization Gap Comparison (Delta = Chicago2025 - NIBRS)
| time_window | model | auc_chicago_2025 | auc_nibrs | delta_auc | f1_chicago_2025 | f1_nibrs | delta_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3d | baseline_persistence | 0.8949656918897455 | 0.9987551692918574 | -0.10378947740211186 | 0.6185737976782753 | 0.9072254966164593 | -0.2886516989381841 |
| 7d | baseline_persistence | 0.9369264744301483 | 0.9982853363943279 | -0.06135886196417961 | 0.6938775510204082 | 0.9420209828823854 | -0.2481434318619773 |
| 3d | logistic_regression | 0.9322119637512866 | 0.836688464694752 | 0.09552349905653457 | 0.7043343152944052 | 0.40419636059737035 | 0.3001379546970348 |
| 7d | logistic_regression | 0.9543560729758562 | 0.8924460164418395 | 0.06191005653401671 | 0.7559309078148146 | 0.5611965915719974 | 0.19473431624281723 |
| 3d | lstm | 0.895126496278289 | 0.7289976845107757 | 0.16612881176751326 | 0.6257224499285613 | 0.13453108863065946 | 0.49119136129790186 |
| 7d | lstm | 0.926907805934731 | 0.922603394778894 | 0.004304411155837018 | 0.6931388603566164 | 0.35084274746294947 | 0.34229611289366696 |
| 3d | random_forest | 0.9297412180093297 | 0.997714804794632 | -0.0679735867853023 | 0.6765502987964632 | 0.9276421471835364 | -0.2510918483870732 |
| 7d | random_forest | 0.9521395293144362 | 0.9968248518466455 | -0.04468532253220925 | 0.7609945205436112 | 0.9507983187619168 | -0.18980379821830562 |

## Output Directories
- `output_timeline_20260308_194946/7d/data_processed`, `output_timeline_20260308_194946/7d/models`, `output_timeline_20260308_194946/7d/metrics`, `output_timeline_20260308_194946/7d/figures`
- `output_timeline_20260308_194946/3d/data_processed`, `output_timeline_20260308_194946/3d/models`, `output_timeline_20260308_194946/3d/metrics`, `output_timeline_20260308_194946/3d/figures`
- Figure 2 (Delta AUC): `output_timeline_20260308_194946/compare_windows/figures/fig2_gap_auc.png`
- Figure 2 (Delta F1): `output_timeline_20260308_194946/compare_windows/figures/fig2_gap_f1.png`
- Compare Overall AUC: `output_timeline_20260308_194946/compare_windows/figures/fig_compare_windows_overall_auc.png`
- Compare Overall F1: `output_timeline_20260308_194946/compare_windows/figures/fig_compare_windows_overall_f1.png`
- Compare Overall Precision: `output_timeline_20260308_194946/compare_windows/figures/fig_compare_windows_overall_precision.png`
- Compare Overall Recall: `output_timeline_20260308_194946/compare_windows/figures/fig_compare_windows_overall_recall.png`