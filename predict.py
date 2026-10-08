"""Generate predictions from an already trained polynomial model."""
from polynomial import design
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = np.load(args.model, allow_pickle=False)
    data = pd.read_csv(args.input)
    if list(data.columns) != model['features'].tolist():
        raise ValueError('Input feature names or order do not match the model.')
    x = data.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Input contains nonfinite values.')
    pred = design(x, model['powers']) @ model['coef'] + model['intercept']
    if not np.isfinite(pred).all():
        raise ValueError('Model produced nonfinite predictions.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y': pred}).to_csv(args.output, index=False)
    print(f'Wrote {len(pred)} predictions to {args.output}')
