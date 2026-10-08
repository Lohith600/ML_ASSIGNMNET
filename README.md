# Assignment 1: Polynomial Regression

Roll number: **BT2024260**. Two independent models predict the net power score
(var1) and thermal anomaly score (var2). All predictors are polynomial terms;
ridge regularization controls coefficient magnitude. No non-polynomial model is used.

## Results

| Problem | Degree | Nonconstant terms | Ridge alpha | 3-fold CV MSE | Holdout MSE | Holdout R2 |
|---|---:|---:|---:|---:|---:|---:|
| var1 | 5 | 461 | 3.162278 | 0.584824 | 0.497835 | 0.961435 |
| var2 | 13 | 559 | 0.316228 | 0.339568 | 0.252532 | 0.994440 |

These scores are measured on the reserved validation splits.
The saved final models are refitted on all 1,000 labeled rows
per problem after the holdout evaluation.

## Run the complete workflow

Use Python 3.12 and install the requirements in your own environment:

```bash
python -m pip install -r requirements.txt
python train.py
python verify.py
python build_report.py
```

Place the four personalized CSV files and `sample_submission.csv` beside the
scripts. The source CSVs are never modified. `train.py` also accepts
`--data-dir`, `--roll`, and `--output-dir`; the report and verification scripts
use the default BT2024260 paths in this assignment workspace.

Training recreates:

- `outputs/BT2024260_pred_var1.csv` and `outputs/BT2024260_pred_var2.csv`
- `outputs/models/var1.npz` and `outputs/models/var2.npz`
- Detailed cross-validation tables, holdout predictions, and `metrics.json`

The report is written to `output/pdf/BT2024260_report.pdf`.

## Run inference without retraining

```bash
python predict.py --model outputs/models/var1.npz --input BT2024260_test_var1.csv --output outputs/BT2024260_pred_var1.csv
python predict.py --model outputs/models/var2.npz --input BT2024260_test_var2.csv --output outputs/BT2024260_pred_var2.csv
```

Each submission has exactly 1,000 rows, one column named `y`, no saved index,
and the same row order as its test file. Negative scores and values outside the
training target range are retained; no unsupported target clipping is applied.

## Method and rationale

1. Check column names, finite numeric values, and duplicate input vectors.
   Inputs already lie in [-1, 1], so no additional input scaling is fitted.
2. Group identical input vectors before splitting. Shuffle unique groups with
   NumPy's random generator, seed 42. Reserve 20% of groups for a final holdout.
   The development/holdout row counts are 800/200 (var1) and 802/198 (var2).
3. Divide the development groups into three folds. Repeated inputs always stay
   in the same fold. Retain all observed responses, including replicate inputs.
4. Search all total degrees 1-10 for var1 and 1-20 for var2. At each degree,
   search 23 ridge penalties `np.logspace(-8, 3, 23)`.
5. Select the degree and alpha with the smallest mean fold MSE. These same
   folds are reused for every candidate to make comparisons consistent.
6. Fit that configuration on the development data and evaluate the holdout
   once. Do not use the holdout to select or revise hyperparameters.
7. Refit the selected configuration on all labeled data, save the model, and
   predict the test inputs.

For each monomial, the sum of feature exponents is at most the chosen degree.
For example, x1^2*x2 has degree 3. The number of nonconstant terms is
`comb(n_features + degree, degree) - 1`. The intercept is handled separately.

The fitted objective is:

```text
sum_i (y_i - intercept - phi(x_i) @ coefficients)^2
    + alpha * sum_j coefficients_j^2
```

There is no division of the squared-error sum by the number of training rows.
The intercept is unpenalized. Polynomial columns and targets are centered using
only the fitted partition. There is no feature standardization after expansion;
the penalty therefore applies to coefficients in the original monomial basis.
This is a modeling choice, not an assumption that all polynomial columns have
equal variance. Fixed bounded inputs avoid exploding raw feature powers.

`polynomial.py` uses the smaller of the feature-space and sample-space Gram
matrices. An eigendecomposition allows the whole alpha path to share one
decomposition per degree and fold. Final fitting uses a positive-definite
linear solve. This keeps the implementation practical even for 8,007 terms at
degree 10 in var1. Both algebraic paths are checked against an independently
constructed augmented least-squares solution in `verify.py`.

## Interpretation and limitations

- Var1's CV MSE reaches its minimum at degree 5, then increases. Adding
  degrees above 5 is not justified by these validation results.
- Var2 benefits from a higher-degree model, with its minimum at degree 13.
  Degree 20 has a higher CV MSE despite greater flexibility.
- Var1 has at least one boundary-valued input in 87.4% of training rows and
  98.2% of test rows. Its holdout boundary MSE is 0.517048, versus 0.334074
  for interior rows. This covariate shift means test performance may differ
  from random holdout performance.
- The CV minimum is a model-selection statistic and can be optimistic. The
  separate holdout provides an additional check, but it is still one split.
- No test-based hyperparameter choices or target clipping were used.
  Test inputs were inspected only for schema and distribution.

## Submission status

The two prediction CSVs and the local report are generated. The report includes
the repository link: [ML_ASSIGNMNET](https://github.com/Lohith600/ML_ASSIGNMNET).
To rebuild it with that link:

```bash
python build_report.py --repo-url https://github.com/Lohith600/ML_ASSIGNMNET
```

The repository URL is also the report builder's default. The `.gitignore` keeps
personalized source datasets out of the repository by default.
