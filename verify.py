"""Independent numerical checks and end-to-end prediction validation."""
from polynomial import design, powers, ridge_path, fit, group_partitions
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(7)
    # Independently solve augmented least squares, including an unpenalized intercept.
    for n, p in [(40, 8), (12, 30)]:
        a, b, y = rng.normal(size=(n, p)), rng.normal(size=(9, p)), rng.normal(size=n)
        alpha = 0.3
        augmented = np.vstack([np.column_stack([np.ones(n), a]),
                               np.column_stack([np.zeros(p), np.sqrt(alpha) * np.eye(p)])])
        target = np.r_[y, np.zeros(p)]
        reference = np.linalg.lstsq(augmented, target, rcond=None)[0]
        expected = np.column_stack([np.ones(len(b)), b]) @ reference
        coef, intercept = fit(a, y, alpha)
        np.testing.assert_allclose(b @ coef + intercept, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(ridge_path(a, y, b, [alpha])[:, 0], expected, rtol=1e-9, atol=1e-9)
    p = powers(2, 2)
    np.testing.assert_array_equal(design(np.array([[2., 3.]]), p), [[2., 3., 4., 6., 9.]])
    for var in ['var1', 'var2']:
        train = pd.read_csv(f'BT2024260_train_{var}.csv')
        x = train.drop(columns='y').to_numpy()
        dev, hold, folds = group_partitions(x)
        as_set = lambda indices: set(map(tuple, x[indices]))
        assert not as_set(dev) & as_set(hold)
        assert len(dev) + len(hold) == len(x)
        for tr, va in folds:
            assert not as_set(tr) & as_set(va)
            assert not set(tr) & set(hold) and not set(va) & set(hold)
        assert sorted(np.concatenate([va for _, va in folds])) == sorted(dev)
        model = np.load(f'outputs/models/{var}.npz', allow_pickle=False)
        if 'model' in model.files and str(model['model']) == 'lasso':
            # Independent optimality check in the standardized feature basis.
            a = design(x, model['powers'])
            scale = a.std(axis=0)
            scale[scale < 1e-12] = 1.0
            z = (a - a.mean(axis=0)) / scale
            residual = train.y.to_numpy() - (a @ model['coef'] + model['intercept'])
            gamma = model['coef'] * scale
            correlations = z.T @ residual / len(x)
            active = gamma != 0
            alpha = float(model['alpha'])
            assert abs(residual.mean()) < 1e-8
            assert np.max(np.abs(correlations[active] - alpha * np.sign(gamma[active]))) < 1e-4
            if (~active).any():
                assert np.max(np.abs(correlations[~active])) <= alpha + 1e-4
        test = pd.read_csv(f'BT2024260_test_{var}.csv')
        pred = design(test.to_numpy(), model['powers']) @ model['coef'] + model['intercept']
        submission = pd.read_csv(f'outputs/BT2024260_pred_{var}.csv')
        assert list(submission.columns) == ['y'] and len(submission) == len(test) == 1000
        assert np.isfinite(submission.y).all()
        np.testing.assert_allclose(pred, submission.y, rtol=1e-12, atol=1e-12)
        assert int(model['powers'].sum(axis=1).max()) == int(model['degree'])
    print('PASS: primal/dual ridge vs independent least squares; Lasso KKT optimality when selected; polynomial terms; grouped splits; saved models; submission schema, count, order, and values.')


if __name__ == '__main__':
    main()
