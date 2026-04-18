# Toy Symbolic Regression

## Problem Statement
Propose a closed-form f(x) that best fits 40 noisy (x, y) points on x ∈ [-4, 4] with Gaussian noise σ=0.03. No sklearn, no fitting loops, no scipy.optimize — symbolic expression only, coefficients tuned by inspection.

## Solution Interface
`orbits/<name>/solution.py` exports `f(x: np.ndarray) -> np.ndarray`. Evaluator at `research/eval/evaluator.py` computes MSE on a held-out clean test set of 400 points on the same range.

## Success Metric
MSE (minimize). Target: MSE < 0.01. Budget: max 2 orbits.
