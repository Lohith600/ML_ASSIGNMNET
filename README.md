# Assignment 1: Polynomial Regression

Roll number: **BT2024260**.

Two separate polynomial models predict Net Power Score (var1) and Thermal
Anomaly Score (var2). Ridge and Lasso were compared for both problems using
the same three development folds. The method with lower average validation
MSE was selected for each problem.

## Model comparison

| Problem | Best Ridge CV MSE | Best Lasso CV MSE | Selected method | Degree | Alpha |
|---|---:|---:|---|---:|---:|
| var1 | 0.584824 | 0.332479 | Lasso | 5 | 0.01 |
| var2 | 0.339568 | 0.368371 | Ridge | 13 | 0.316228 |

Lasso reduces var1's cross-validation MSE by about **43.1%**. Ridge remains
better for var2 within the tested grids.

| Problem | Selected-model holdout MSE | Holdout R2 |
|---|---:|---:|
| var1 | 0.371759 | 0.971201 |
| var2 | 0.252532 | 0.994440 |

The existing holdout was already viewed during the earlier Ridge analysis.
It is reused as a diagnostic, not a fresh independent evaluation. Model
selection uses only development-fold CV scores. Final models are refitted
on all 1,000 training rows for each problem.

## Run the project

Use Python 3.12. Place the four personalized dataset CSVs and
`sample_submission.csv` beside the scripts, then run:

```bash
python -m pip install -r requirements.txt
python train.py
python verify.py
python build_report.py
```

`train.py` runs both Ridge searches and both Lasso comparisons before choosing
models. `--data-dir`, `--roll`, and `--output-dir` are supported. The report and
verification scripts use the default BT2024260 paths. An optional
`--reuse-lasso` flag reuses existing comparison results; use it only when the
datasets and comparison settings are unchanged. The default reruns the search.

Run inference without retraining:

```bash
python predict.py --model outputs/models/var1.npz --input BT2024260_test_var1.csv --output outputs/BT2024260_pred_var1.csv
python predict.py --model outputs/models/var2.npz --input BT2024260_test_var2.csv --output outputs/BT2024260_pred_var2.csv
```

## Method

1. Check columns, missing values, and repeated inputs. All inputs lie in [-1, 1].
2. Keep identical input vectors together in every split. Seed 42 reserves 20%
   of input groups for holdout, leaving 800 development rows for var1 and 802
   for var2. Split those development groups into three folds.
3. Create all polynomial terms up to each candidate total degree. For example,
   x1^2*x2 has total degree 3. Test degrees 1-10 for var1 and 1-20 for var2.
4. For Ridge, try 23 alpha values from 1e-8 to 1e3. For Lasso, try 19 values
   from 0.001 to 1 for var1 and 25 from 0.0001 to 1 for var2. Grids are
   logarithmically spaced. All candidates use identical folds.
5. Select the method, degree, and alpha with the lowest mean fold MSE. Check
   holdout performance, then refit on all labeled rows and predict test rows.

Ridge centers polynomial columns using only its fitting partition. Lasso
centers and scales columns to unit standard deviation using only its fitting
partition. Constant columns use scale 1. Both have an unpenalized intercept.

The objectives use different conventions:

```text
Ridge: SSE + alpha * sum(coefficient^2)
Lasso: SSE/(2*n) + alpha * sum(abs(coefficient))
```

SSE is the sum of squared errors and n is the fitting row count. Ridge penalizes
raw polynomial coefficients; Lasso penalizes standardized coefficients. Their
alpha values should not be compared directly.

`polynomial.py` implements Ridge with NumPy/SciPy. `compare_lasso.py` uses
scikit-learn's LassoLars to compute the piecewise-linear Lasso path and
interpolates it at the alpha grid. An independent primal/dual objective check
requires relative gap <= 1.01e-6 for eligible CV candidates. Final Lasso fits
use coordinate descent with a relative dual-gap threshold of 1e-7. Saved
Lasso coefficients are converted back to the original feature basis, so
`predict.py` does not need scikit-learn or a separate scaler at inference time.

The initial coordinate-descent search was replaced by the LARS path solver to
resolve slow convergence with strongly correlated, high-degree terms. The
reported comparison files contain the completed LARS searches.

## Files and checks

- `outputs/BT2024260_pred_var1.csv` and `outputs/BT2024260_pred_var2.csv`:
  1,000 predictions each, one `y` column, no index, original test-row order.
- `outputs/models/`: saved coefficients, exponents, intercepts, and settings.
- `outputs/*_cv_results.csv`: all searched settings and fold errors; files
  without `lasso` in the name record the Ridge search.
- `outputs/*_model_comparison.json`: best Ridge/Lasso settings and the winner.
- `outputs/metrics.json`: selected-model evaluation results.
- `outputs/ridge_baseline_metrics.json`: original Ridge-only results for reference.
- `output/pdf/BT2024260_report.pdf`: the assignment report.

`verify.py` checks Ridge against an independent least-squares solution and
checks the selected Lasso coefficients against its optimality conditions.
It also checks polynomial terms, split separation, serialized predictions,
finite values, CSV columns, row count, and row order.

Var1's test inputs are more concentrated at the boundaries than its training
inputs (98.2% versus 87.4% have at least one input at -1 or 1). This may affect
prediction accuracy. No target clipping or test-based model selection is used.
Original datasets are unchanged and are excluded from Git by default.

## Repository

[ML_ASSIGNMNET](https://github.com/Lohith600/ML_ASSIGNMNET)

The report includes this link by default. To use another URL:

```bash
python build_report.py --repo-url https://github.com/Lohith600/ML_ASSIGNMNET
```
