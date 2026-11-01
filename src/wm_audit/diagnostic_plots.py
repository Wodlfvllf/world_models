"""Headless, offline plots; no dashboards or extra servers needed."""
import html
from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _gallery(output, title, figures, caveat):
    cards = "".join(f'<section><h2>{html.escape(label)}</h2><img src="{name}" alt="{html.escape(label)}"></section>'
                    for name, label in figures)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font:16px system-ui;background:#101827;color:#e5edf8;margin:0;padding:32px;max-width:1250px;margin:auto}}
section{{background:#1b293e;padding:20px;margin:24px 0;border-radius:16px}}
img{{width:100%;height:auto;border-radius:8px}}p{{line-height:1.6;color:#c5d6ec}}a{{color:#8edbff}}
</style><h1>{html.escape(title)}</h1><p>{html.escape(caveat)}</p>
<p><a href="report.json">Structured results and provenance</a></p>{cards}</html>'''
    (Path(output) / "index.html").write_text(content, encoding="utf-8")


def readout_plots(output, scatter, candidates):
    plt = _pyplot()
    output = Path(output)
    names = ("Pusher X", "Pusher Y", "Block X", "Block Y", "sin(angle)", "cos(angle)")
    figures, arrays = [], {}
    for appearance, (actual_parts, predicted_parts) in scatter.items():
        actual, predicted = np.concatenate(actual_parts), np.concatenate(predicted_parts)
        arrays[f"{appearance}_actual"], arrays[f"{appearance}_predicted"] = actual, predicted
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for i, ax in enumerate(axes.flat):
            ax.scatter(actual[:, i], predicted[:, i], s=6, alpha=.25, rasterized=True)
            lo, hi = min(actual[:, i].min(), predicted[:, i].min()), max(actual[:, i].max(), predicted[:, i].max())
            ax.plot([lo, hi], [lo, hi], "--", color="#e76f51")
            ax.set(title=names[i], xlabel="Simulator truth", ylabel="Readout prediction")
            ax.grid(alpha=.2)
        fig.suptitle(f"Actual-image readout: {appearance} (all diagnostic samples)")
        name = f"scatter-{appearance}.png"
        fig.savefig(output / name, dpi=120)
        plt.close(fig)
        figures.append((name, f"Predicted versus true: {appearance}"))
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        angle = np.arctan2(predicted[:, 4], predicted[:, 5]) - np.arctan2(actual[:, 4], actual[:, 5])
        error_angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle))) * 180 / np.pi
        error_angle[np.linalg.norm(predicted[:, 4:], axis=1) < 1e-8] = 180
        errors = [np.abs(predicted[:, i] - actual[:, i]) for i in range(4)] + [error_angle, np.linalg.norm(predicted[:, 4:], axis=1)]
        for ax, values, label in zip(axes.flat, errors, (*names[:4], "Angular error (degrees)", "Predicted sin/cos norm")):
            ax.hist(values, bins=40, color="#2878b5")
            ax.set(title=label, xlabel="Absolute error (native units)" if label in names else label, ylabel="Samples")
        name = f"errors-{appearance}.png"
        fig.savefig(output / name, dpi=120)
        plt.close(fig)
        figures.append((name, f"Error distributions: {appearance}"))
    np.savez_compressed(output / "scatter-data.npz", **arrays)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for grid in sorted({r["grid"] for r in candidates}):
        rows = sorted((r for r in candidates if r["grid"] == grid), key=lambda r: r["alpha"])
        ax.plot([r["alpha"] for r in rows], [r["validation_scaled_loss"] for r in rows], "o-", label=f"grid {grid}")
    ax.set(xscale="log", xlabel="Ridge alpha (sum-squared objective)", ylabel="Original validation scaled loss",
           title="Readout selection — no pilot or shifted images used")
    ax.legend()
    fig.savefig(output / "selection.png", dpi=120)
    plt.close(fig)
    figures.insert(0, ("selection.png", "Development-only model selection"))
    _gallery(output, "Physical readout diagnostic", figures,
             "Exploratory, not confirmation. Fits use actual images only. Positions are native simulator units, not resized-image pixels. Scatter rows are correlated within roots; confidence intervals in the report resample roots, not individual frames.")


def prediction_plots(output, summary):
    plt = _pyplot()
    figures = []
    for appearance in sorted({row["appearance"] for row in summary}):
        rows = sorted((r for r in summary if r["appearance"] == appearance), key=lambda r: r["horizon"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for method in ("native", "persistence", "zero_future_actions"):
            points = [r["metrics"][method] for r in rows]
            axes[0].plot([r["horizon"] for r in rows], [p["estimate"] for p in points], "o-", label=method)
            axes[0].fill_between([r["horizon"] for r in rows], [p["ci95"][0] for p in points], [p["ci95"][1] for p in points], alpha=.15)
        for contrast in ("native_minus_persistence", "native_minus_zero_future_actions"):
            axes[1].plot([r["horizon"] for r in rows], [r[contrast]["estimate"] for r in rows], "o-", label=contrast)
            axes[1].fill_between([r["horizon"] for r in rows], [r[contrast]["ci95"][0] for r in rows], [r[contrast]["ci95"][1] for r in rows], alpha=.15)
        axes[1].axhline(0, color="grey", linestyle="--")
        for ax in axes:
            ax.set(xlabel="Model forecast steps (5 primitive controls each)", ylabel="Latent MSE", title=appearance)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        name = f"prediction-{appearance}.png"
        fig.savefig(Path(output) / name, dpi=120)
        plt.close(fig)
        figures.append((name, f"Frozen prediction: {appearance}"))
    _gallery(output, "Frozen prediction diagnostic", figures,
             "Shaded bands: pointwise 95% root/episode bootstrap intervals. Negative paired contrasts favor native prediction. Raw latent MSE is comparable within a model only. Recorded pretraining data is not held-out world-model generalization. No physical readout is used.")