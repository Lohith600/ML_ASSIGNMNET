"""Compare standardized polynomial Lasso on the original development folds."""
from polynomial import design, powers, group_partitions
import sys
from pathlib import Path
if Path('.deps').exists():
    sys.path.insert(0, str(Path('.deps').resolve()))
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLars, Lasso
from scipy.interpolate import interp1d


def relative_dual_gap(z, yc, coefs, alphas):
    """Independent primal/dual objective check for standardized Lasso solutions."""
    residual = yc[:, None] - z @ coefs
    n = len(yc)
    dual_scale = np.maximum(1.0, np.max(np.abs(z.T @ residual), axis=0) / (n * alphas))
    theta = residual / (n * dual_scale)
    primal = np.sum(residual ** 2, axis=0) / (2 * n) + alphas * np.sum(np.abs(coefs), axis=0)
    dual = yc @ theta - n * np.sum(theta ** 2, axis=0) / 2
    return np.maximum(primal - dual, 0) / np.var(yc)


def fit_lasso(a, y, alpha):
    mean, scale = a.mean(axis=0), a.std(axis=0)
    scale[scale < 1e-12] = 1.0
    model = Lasso(alpha=alpha, max_iter=100000, tol=1e-7)
    model.fit((a - mean) / scale, y)
    if model.dual_gap_ > 1e-7 * np.var(y):
        raise RuntimeError('Final Lasso failed convergence check')
    coef = model.coef_ / scale
    return coef, float(model.intercept_ - mean @ coef)


def compare(root=Path('.'), roll='BT2024260', out=Path('outputs'), var='var1'):
    data = pd.read_csv(root / f'{roll}_train_{var}.csv')
    x, y = data.drop(columns='y').to_numpy(), data.y.to_numpy()
    _, _, folds = group_partitions(x)
    max_degree = 10 if var == 'var1' else 20
    exponents = powers(x.shape[1], max_degree)
    full = design(x, exponents)
    alphas = np.logspace(0, -3, 19) if var == 'var1' else np.logspace(0, -4, 25)
    records = []
    for degree in range(1, max_degree + 1):
        count = int((exponents.sum(axis=1) <= degree).sum())
        a = full[:, :count]
        scores, gaps, nonzeros = [], [], []
        for tr, va in folds:
            mean, scale = a[tr].mean(axis=0), a[tr].std(axis=0)
            scale[scale < 1e-12] = 1.0
            z = np.asfortranarray((a[tr] - mean) / scale)
            yc = y[tr] - y[tr].mean()
            model = LassoLars(alpha=float(alphas[-1]), fit_intercept=False, max_iter=10000)
            model.fit(z, yc)
            if model.alphas_[-1] > alphas[-1] * 1.001:
                raise RuntimeError('Lasso path stopped before the requested minimum alpha.')
            path = interp1d(model.alphas_[::-1], model.coef_path_[:, ::-1], axis=1,
                            bounds_error=False, fill_value=(model.coef_path_[:, -1], np.zeros(count)))
            coefs = path(alphas)
            pred = ((a[va] - mean) / scale) @ coefs + y[tr].mean()
            scores.append(np.mean((y[va, None] - pred) ** 2, axis=0))
            gaps.append(relative_dual_gap(z, yc, coefs, alphas))
            nonzeros.append(np.count_nonzero(coefs, axis=0))
        for k, alpha in enumerate(alphas):
            fs = [float(s[k]) for s in scores]
            records.append(dict(model='lasso', degree=degree, features=count, alpha=float(alpha),
                                cv_mse=float(np.mean(fs)), cv_std=float(np.std(fs)),
                                max_relative_dual_gap=float(max(g[k] for g in gaps)),
                                mean_nonzero=float(np.mean([nz[k] for nz in nonzeros])),
                                **{f'fold_{j+1}_mse': s for j, s in enumerate(fs)}))
        pd.DataFrame(records).to_csv(out / f'{var}_lasso_cv_results.csv', index=False)
        best = min(records[-len(alphas):], key=lambda r: r['cv_mse'])
        print(f"{var} degree {degree}: MSE={best['cv_mse']:.6f}, alpha={best['alpha']:.6g}, nonzero={best['mean_nonzero']:.1f}, gap={best['max_relative_dual_gap']:.3g}", flush=True)
    candidates = [r for r in records if r['max_relative_dual_gap'] <= 1.01e-6]
    best = min(candidates, key=lambda r: r['cv_mse'])
    ridge = pd.read_csv(out / f'{var}_cv_results.csv')
    ridge_best = ridge.loc[ridge.cv_mse.idxmin()].to_dict()
    result = dict(lasso=best, ridge=ridge_best, winner='lasso' if best['cv_mse'] < ridge_best['cv_mse'] else 'ridge')
    (out / f'{var}_model_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--var', choices=['var1', 'var2'], default='var1')
    args = parser.parse_args()
    compare(var=args.var)
