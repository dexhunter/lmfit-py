#!/usr/bin/env python3
"""Baseline wall time for one evaluation of a many-peak composite model.

``Model.make_funcargs`` walks every parameter for every component. A sum of
prefixed peak models, which is how lmfit builds multi-peak spectra, therefore
does work that grows with the square of the peak count. This harness times
``Model.eval`` on that workload and does not change lmfit.

One peak's sigma is constrained to the first peak, so expression parameters are
part of the measured path.

Protocol
--------
BLAS and OpenMP threads are pinned to 1. After one warmup evaluation, five
rounds are timed. Each round is the median of seven evaluations.
``baseline_noise_pct`` is the full range of those five medians as a percent
of their median: ``(max - min) / median * 100``.
"""

import gc
import json
import os
import statistics
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from lmfit.models import GaussianModel

N_PEAKS = 80
N_X = 2000
N_ROUNDS = 5
N_INNER = 7


def build_model():
    """Sum of prefixed Gaussian peaks, as in tests/test_manypeaks_speed.py."""
    model = None
    for i in range(N_PEAKS):
        peak = GaussianModel(prefix=f'g{i}_')
        model = peak if model is None else model + peak
    params = model.make_params()
    for i in range(N_PEAKS):
        params[f'g{i}_amplitude'].set(value=1.0)
        params[f'g{i}_center'].set(value=-1.0 + 2.0 * i / (N_PEAKS - 1))
        params[f'g{i}_sigma'].set(value=0.08)
        if i:
            params[f'g{i}_sigma'].set(expr='g0_sigma')
    x = np.linspace(-2.0, 2.0, N_X)
    return model, params, x


def median_eval(model, params, x):
    samples = []
    for _ in range(N_INNER):
        started = time.perf_counter()
        model.eval(params, x=x)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main():
    model, params, x = build_model()
    warmup = np.asarray(model.eval(params, x=x), dtype=np.float64)
    rounds = []
    for _ in range(N_ROUNDS):
        gc.collect()
        rounds.append(median_eval(model, params, x))
    baseline = statistics.median(rounds)
    noise_pct = (max(rounds) - min(rounds)) / baseline * 100.0
    report = {
        'metric_kind': 'wall_time_s',
        'baseline_value': baseline,
        'baseline_noise_pct': noise_pct,
        'baseline_samples_s': rounds,
        'n_peaks': N_PEAKS,
        'n_x': N_X,
        'rounds': N_ROUNDS,
        'inner_repeats': N_INNER,
        'checksum': float(warmup.sum()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
