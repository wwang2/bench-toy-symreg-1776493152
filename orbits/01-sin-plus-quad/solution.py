"""Closed-form fit for the toy symbolic-regression problem.

By inspection of research/eval/train_data.csv on x in [-4, 4]:
  - y(-4) ~= 2.37 and y(+4) ~= 0.85 are asymmetric -> odd component (sin-like).
  - Midpoint average (y(-4)+y(+4))/2 ~= 1.61 ~= 0.1 * 4^2 -> quadratic envelope 0.1 x^2.
  - Half-difference (y(-4)-y(+4))/2 ~= 0.76 ~= sin(4), confirming a sin(x) component.
Hence: f(x) = sin(x) + 0.1 * x^2 (coefficients set by eyeball, no fitting loop).
"""

import numpy as np


def f(x: np.ndarray) -> np.ndarray:
    return np.sin(x) + 0.1 * x**2
