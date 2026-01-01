import numpy as np
import matplotlib.pyplot as plt


def plot_all_metrics_with_ci(metrics: dict, titles: dict, ncols=3, figsize=(16, 8), labels=None):
    """
    Plot one or multiple runs per metric with confidence intervals.

    Args:
        metrics: dict of metric_name -> metric_dict OR list/tuple of metric_dicts.
                 Each metric_dict follows EvaluateRetrievalCI output.
                 If a list is provided, multiple curves are drawn in the same subplot.
        titles: dict of metric_name -> readable plot title.
        labels: optional list of labels used when a metric has multiple curves.
                If None, defaults to "run 1", "run 2", ...
    """
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, metric_name in zip(axes, metric_names):
        metric_value = metrics[metric_name]
        if isinstance(metric_value, (list, tuple)):
            metric_dicts = list(metric_value)
        else:
            metric_dicts = [metric_value]

        # labels per metric
        local_labels = labels or [f"run {i+1}" for i in range(len(metric_dicts))]

        y_min, y_max = np.inf, -np.inf
        for idx, metric_dict in enumerate(metric_dicts):
            items = []
            for name, stats in metric_dict.items():
                k = int(name.split("@")[1])
                items.append((k, stats))
            items.sort(key=lambda x: x[0])

            ks = np.array([k for k, _ in items])
            means = np.array([s["mean"] for _, s in items])
            ci_low = np.array([s["ci"][0] for _, s in items])
            ci_high = np.array([s["ci"][1] for _, s in items])

            yerr = np.vstack([means - ci_low, ci_high - means])

            ax.errorbar(
                ks,
                means,
                yerr=yerr,
                fmt="o-",
                capsize=4,
                elinewidth=1,
                markersize=5,
                label=local_labels[idx] if len(local_labels) > idx else None,
            )

            y_min = min(y_min, ci_low.min())
            y_max = max(y_max, ci_high.max())

        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)

        ax.set_ylim(y_min * 0.97 - 0.01, y_max * 1.01 + 0.01)
        ax.set_title(titles.get(metric_name, metric_name))
        ax.set_xlabel("Cutoff k")
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle="--", alpha=0.5)

        if len(metric_dicts) > 1:
            ax.legend()

    for i in range(len(metric_names), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Retrieval Metrics with Confidence Intervals", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
