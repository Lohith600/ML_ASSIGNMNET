"""Select polynomial models by grouped CV and write verified predictions."""
from polynomial import design, powers, ridge_path, fit, metrics, group_partitions
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def run(root, roll, out, reuse_lasso=False):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    sample = pd.read_csv(root / 'sample_submission.csv')
    if list(sample.columns) != ['y']:
        raise ValueError('Expected sample submission to contain only y.')
    summary = {}
    alphas = np.logspace(-8, 3, 23)
    for var, n_features, max_degree in [('var1', 6, 10), ('var2', 3, 20)]:
        train = pd.read_csv(root / f'{roll}_train_{var}.csv')
        test = pd.read_csv(root / f'{roll}_test_{var}.csv')
        names = [f'x{i}' for i in range(1, n_features + 1)]
        if list(train.columns) != names + ['y'] or list(test.columns) != names:
            raise ValueError(f'Unexpected schema for {var}')
        x, y, xt = train[names].to_numpy(), train.y.to_numpy(), test.to_numpy()
        if not all(np.isfinite(z).all() for z in [x, y, xt]):
            raise ValueError('Nonfinite data found.')
        dev, hold, folds = group_partitions(x)
        exponents = powers(n_features, max_degree)
        full = design(x, exponents)
        records = []
        print(f'{var}: {len(dev)} development, {len(hold)} holdout; {len(exponents)} maximum features', flush=True)
        for degree in range(1, max_degree + 1):
            count = int((exponents.sum(axis=1) <= degree).sum())
            a = full[:, :count]
            fold_scores = []
            for tr, va in folds:
                predictions = ridge_path(a[tr], y[tr], a[va], alphas)
                fold_scores.append(np.mean((predictions - y[va, None]) ** 2, axis=0))
            for k, alpha in enumerate(alphas):
                scores = [float(s[k]) for s in fold_scores]
                records.append({'degree': degree, 'alpha': float(alpha), 'features': count,
                                'cv_mse': float(np.mean(scores)), 'cv_std': float(np.std(scores)),
                                **{f'fold_{i+1}_mse': s for i, s in enumerate(scores)}})
            best_here = min(records[-len(alphas):], key=lambda r: r['cv_mse'])
            print(f"  degree {degree:2d}: CV MSE {best_here['cv_mse']:.6g}, alpha {best_here['alpha']:.3g}", flush=True)
        cv = pd.DataFrame(records)
        cv.to_csv(out / f'{var}_cv_results.csv', index=False)
        best = min(records, key=lambda r: (r['cv_mse'], r['degree']))
        best = dict(best, model='ridge')
        fitter = fit
        if var in ('var1', 'var2'):
            from compare_lasso import compare, fit_lasso
            # Reuse is an explicit option for this unchanged dataset's recorded search.
            comparison = (json.loads((out / f'{var}_model_comparison.json').read_text())
                          if reuse_lasso else compare(root, roll, out, var))
            if comparison['winner'] == 'lasso':
                best = comparison['lasso']
                fitter = fit_lasso
        count = best['features']
        a = full[:, :count]
        coef, intercept = fitter(a[dev], y[dev], best['alpha'])
        hold_pred = a[hold] @ coef + intercept
        hold_metrics = metrics(y[hold], hold_pred)
        train_metrics = metrics(y[dev], a[dev] @ coef + intercept)
        baseline = metrics(y[hold], np.full(len(hold), y[dev].mean()))
        boundary = (np.abs(x[hold]) >= 0.999999).any(axis=1)
        pd.DataFrame({'source_row_zero_based': hold, 'y_true': y[hold], 'y_pred': hold_pred,
                      'residual': y[hold] - hold_pred, 'boundary': boundary}).to_csv(out / f'{var}_holdout.csv', index=False)
        coef, intercept = fitter(a, y, best['alpha'])
        final_exponents = exponents[:count]
        pred = design(xt, final_exponents) @ coef + intercept
        if len(pred) != len(sample) or not np.isfinite(pred).all():
            raise ValueError('Invalid prediction shape or values.')
        prediction_path = out / f'{roll}_pred_{var}.csv'
        pd.DataFrame({'y': pred}).to_csv(prediction_path, index=False)
        np.savez_compressed(out / 'models' / f'{var}.npz', powers=final_exponents,
                            coef=coef, intercept=intercept, features=np.asarray(names),
                            degree=best['degree'], alpha=best['alpha'], model=best['model'])
        # Verify serialization and CSV row/column fidelity.
        saved = np.load(out / 'models' / f'{var}.npz', allow_pickle=False)
        restored = design(xt, saved['powers']) @ saved['coef'] + saved['intercept']
        assert np.allclose(restored, pred, rtol=1e-12, atol=1e-12)
        written = pd.read_csv(prediction_path)
        assert list(written.columns) == ['y'] and written.shape == sample.shape
        assert np.allclose(written.y, pred, rtol=1e-12, atol=1e-12)
        summary[var] = {'selected': best, 'holdout': hold_metrics, 'development_fit': train_metrics,
                        'final_nonzero_terms': int(np.count_nonzero(coef)),
                        'mean_baseline_holdout': baseline, 'n_train': len(x), 'n_test': len(xt),
                        'n_development': len(dev), 'n_holdout': len(hold),
                        'duplicate_training_inputs': int(train.duplicated(subset=names).sum()),
                        'train_boundary_fraction': float((np.abs(x) >= 0.999999).any(axis=1).mean()),
                        'test_boundary_fraction': float((np.abs(xt) >= 0.999999).any(axis=1).mean()),
                        'boundary_holdout': metrics(y[hold][boundary], hold_pred[boundary]),
                        'interior_holdout': metrics(y[hold][~boundary], hold_pred[~boundary]),
                        'prediction_min': float(pred.min()), 'prediction_max': float(pred.max())}
        print(f'{var} selected: {best}; holdout: {hold_metrics}', flush=True)
        (out / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--roll', default='BT2024260')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--reuse-lasso', action='store_true', help='Reuse the recorded Lasso search for the unchanged dataset.')
    args = parser.parse_args()
    run(args.data_dir, args.roll, args.output_dir, args.reuse_lasso)
