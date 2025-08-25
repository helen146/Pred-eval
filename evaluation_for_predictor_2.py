"""
Evaluation of Machine Learning Predictors of Extreme Events in Complex Dynamical Systems

Reference:
Guth, S. & Sapsis, T.P. (2021). Machine Learning Predictors of Extreme Events Occurring in Complex Dynamical Systems.
Nature Communications, 12, 4868. https://doi.org/10.1038/s41467-021-25113-7

Implements precision-recall curves, adjusted AUC, optimal threshold search, and volume under PR-Rate surface,
as recommended in the above paper for evaluating predictors of rare/extreme events.

Author: helen146
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
import os
import warnings

def evaluate_at_threshold(a: np.ndarray, b: np.ndarray, a_thresh: float, plot: bool = False) -> dict:
    """
    Compute precision–recall metrics for a fixed extreme-event cutoff.
    """
    # Input validation
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Inputs 'a' and 'b' must be numpy arrays")
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")
    if not np.isfinite(a_thresh):
        raise ValueError("Threshold must be a finite number")

    # Remove NaNs for robustness
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    # binarize
    y_true = (a > a_thresh).astype(int)
    q = y_true.mean()  # Event rate

    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, b)

    # F1 per point
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    # area under PR
    pr_auc = auc(recall, precision)
    adj_auc = pr_auc - q

    # pick best F1 (ignore the last trivial point)
    best_idx = np.nanargmax(f1[:-1])
    best_threshold = thresholds[best_idx]
    best_f1        = f1[best_idx]
    best_precision = precision[best_idx]
    best_recall    = recall[best_idx]

    results = {
        'a_thresh':      a_thresh,
        'q':             q,
        'precision':     precision,
        'recall':        recall,
        'thresholds':    thresholds,
        'f1':            f1,
        'best_threshold':best_threshold,
        'best_f1':       best_f1,
        'best_precision':best_precision,
        'best_recall':   best_recall,
        'auc':           pr_auc,
        'adjusted_auc':  adj_auc
    }

    if plot:
        plot_precision_recall_curve(recall, precision, best_recall, best_precision, best_f1, q, a_thresh)

    return results

def sweep_extreme_thresholds(a, b, percentiles):
    """
    Find the percentile cutoff that maximizes adjusted AUC = AUC - q.
    """
    best_adj_auc = -np.inf
    best_stats = None
    for p in percentiles:
        a_thresh = np.percentile(a, p)
        res = evaluate_at_threshold(a, b, a_thresh, plot=False)
        if res['adjusted_auc'] > best_adj_auc:
            best_adj_auc = res['adjusted_auc']
            best_stats = {
                'a_thresh':   a_thresh,
                'adjusted_auc': res['adjusted_auc'],
                'precision':  res['precision'],
                'recall':     res['recall'],
                'thresholds': res['thresholds']
            }
    return best_stats

def compute_prr_surface_volume(a, b, percentiles):
    """
    Compute V = ∫ α(q) dq by sweeping `a`-percentiles → event rates q → AUC α(q).
    Returns V, qs_sorted, aucs_sorted, thresholds_sorted
    """
    thresholds = np.percentile(a, percentiles)
    qs, aucs = [], []
    for t in thresholds:
        res = evaluate_at_threshold(a, b, a_thresh=t, plot=False)
        qs.append(res['q'])
        aucs.append(res['auc'])

    qs = np.array(qs)
    aucs = np.array(aucs)
    order = np.argsort(qs)
    qs_sorted = qs[order]
    aucs_sorted = aucs[order]
    thresholds_sorted = thresholds[order]

    V = np.trapz(aucs_sorted, qs_sorted)
    return V, qs_sorted, aucs_sorted, thresholds_sorted

def find_max_adjusted_auc(a, b, percentiles):
    """
    Sweep percentiles and pick the threshold t that maximizes α(q) - q.
    """
    best_val = -np.inf
    best = None
    for p in percentiles:
        t = np.percentile(a, p)
        res = evaluate_at_threshold(a, b, a_thresh=t, plot=False)
        adj = res['adjusted_auc']
        if adj > best_val:
            best_val = adj
            best = {
                'a_thresh':      t,
                'q':             res['q'],
                'auc':           res['auc'],
                'adjusted_auc':  adj,
                'precision':     res['precision'],
                'recall':        res['recall'],
                'thresholds':    res['thresholds']
            }
    return best

def save_results_csv(filename, qs, aucs, thresholds):
    """
    Save event rates, AUCs, and thresholds to CSV for downstream analysis.
    """
    df = pd.DataFrame({
        'event_rate_q': qs,
        'auc_alpha_q': aucs,
        'threshold_a': thresholds
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot_auc_vs_event_rate(qs, aucs, V, percentiles=None):
    """
    Plot AUC vs. Event Rate with volume annotation.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(qs, aucs, '-o', label='AUC vs. Event Rate')
    plt.xlabel('Event rate q')
    plt.ylabel('AUC α(q)')
    plt.title(f'AUC vs. Event Rate (V = {V:.4f})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(recall, precision, best_recall, best_precision, best_f1, q, a_thresh):
    """
    Plot the precision-recall curve at the optimal extreme-event threshold.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='PR Curve')
    plt.scatter(best_recall, best_precision, color='red',
                label=f"Best F1 = {best_f1:.3f}")
    plt.hlines(q, 0, 1, colors='k', linestyles='--',
               label=f'Baseline = {q:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve at a_thresh = {a_thresh:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_actual_histogram(a, a_thresh, bins=50, title=None):
    """
    Plot a histogram of Actual values with the extreme event threshold marked.
    """
    plt.figure(figsize=(8,6))
    plt.hist(a, bins=bins, color='skyblue', edgecolor='k', alpha=0.7, label='Actual values')
    plt.axvline(a_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold (a* = {a_thresh:.3f})')
    plt.xlabel('Actual value')
    plt.ylabel('Count')
    plt.title(title or f'Histogram of Actual Values with Extreme Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(actual_path, predicted_path, percentiles=None, output_dir=None):
    """
    Main evaluation pipeline using Guth & Sapsis (2021) metrics.
    """
    try:
        if percentiles is None:
            percentiles = np.linspace(80, 99.5, 100)
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load and validate data
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"Actual values file not found: {actual_path}")
        if not os.path.exists(predicted_path):
            raise FileNotFoundError(f"Predicted values file not found: {predicted_path}")

        # Load arrays and print shapes
        print("\nLoading arrays...")
        Actual = np.load(actual_path)
        predicted = np.load(predicted_path)
        
        print("Original shapes:")
        print("Actual shape:", Actual.shape)
        print("Predicted shape:", predicted.shape)

        # Fix for column/row vectors
        Actual = Actual.flatten()
        predicted = predicted.flatten()
        
        print("\nAfter flattening:")
        print("Actual shape:", Actual.shape)
        print("Predicted shape:", predicted.shape)

        # Match array lengths if needed
        if Actual.shape != predicted.shape:
            print("\nTruncating arrays to match lengths...")
            minlen = min(len(Actual), len(predicted))
            Actual = Actual[:minlen]
            predicted = predicted[:minlen]
            print("Final shapes:", Actual.shape, predicted.shape)

        # Check for NaN values
        if np.isnan(Actual).any() or np.isnan(predicted).any():
            warnings.warn("Input data contains NaN values", UserWarning)

        print("\nStarting evaluation...\n")

        # 1) Volume under PR-Rate surface
        V, qs, aucs, thr = compute_prr_surface_volume(Actual, predicted, percentiles)
        print(f"Volume under PR-Rate surface V = {V:.4f}")

        if output_dir:
            save_results_csv(os.path.join(output_dir, "prr_surface_results.csv"), 
                           qs, aucs, thr)

        # 2) Maximum adjusted AUC α*
        best = find_max_adjusted_auc(Actual, predicted, percentiles)
        print(f"Optimal extreme threshold a^ = {best['a_thresh']:.4f}")
        print(f"Event rate q         = {best['q']:.4f}")
        print(f"AUC α(q)             = {best['auc']:.4f}")
        print(f"Max adjusted AUC α*  = {best['adjusted_auc']:.4f}")

        # 3) Visualize AUC vs q
        plot_auc_vs_event_rate(qs, aucs, V)

        # 4) Final PR curve at a^
        eval_best = evaluate_at_threshold(
            Actual, predicted,
            a_thresh=best['a_thresh'],
            plot=True
        )

        # 5) Histogram of Actual values with threshold marked
        plot_actual_histogram(Actual, best['a_thresh'])
        
        print("\nFinal metrics:")
        print(f"Best F1: {eval_best['best_f1']:.4f}")
        print(f"Precision: {eval_best['best_precision']:.4f}")
        print(f"Recall: {eval_best['best_recall']:.4f}")

        if output_dir:
            df_best = pd.DataFrame({
                'precision': eval_best['precision'],
                'recall': eval_best['recall'],
                'thresholds': np.append(eval_best['thresholds'], np.nan),
                'f1': eval_best['f1']
            })
            df_best.to_csv(os.path.join(output_dir, "pr_curve_best.csv"), index=False)
            print(f"\nResults saved to {output_dir}")

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    actual_path = os.path.join(base_dir, "Actual.npy")
    predicted_path = os.path.join(base_dir, "Predicted_tau_20.npy")
    output_dir = os.path.join(base_dir, "evaluation_results")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set percentiles
    percentiles = np.linspace(80, 99.5, 100)

    # Run main function with improved error handling
    try:
        main(actual_path, predicted_path, percentiles=percentiles, output_dir=output_dir)
    except Exception as e:
        print(f"\nError running evaluation: {str(e)}")
        print("\nPlease ensure:")
        print("1. Both 'Actual.npy' and 'Predicted_tau_20.npy' exist in the script directory")
        print("2. The arrays have compatible shapes")
        print("3. The arrays contain valid numerical data")