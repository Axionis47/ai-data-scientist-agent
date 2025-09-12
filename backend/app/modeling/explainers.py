from pathlib import Path
from typing import Dict, Any


def add_binary_curves(
    chosen, Xte, yte, plots_dir: Path, base_url: str, explain: Dict[str, Any]
):
    from sklearn.metrics import roc_curve, precision_recall_curve
    import matplotlib.pyplot as plt

    proba = chosen.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, proba)
    prec, rec, _ = precision_recall_curve(yte, proba)
    # ROC
    fig = plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", color="#94a3b8")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    p = plots_dir / "roc_curve.png"
    fig.savefig(p)
    plt.close(fig)
    # PR
    fig = plt.figure(figsize=(4, 3))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    p = plots_dir / "pr_curve.png"
    fig.savefig(p)
    plt.close(fig)
    explain["roc"] = f"{base_url}/roc_curve.png"
    explain["pr"] = f"{base_url}/pr_curve.png"


def add_regression_diagnostics(
    chosen, Xte, yte, plots_dir: Path, base_url: str, explain: Dict[str, Any]
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    import pandas as pd

    resid = yte - pd.Series(chosen.predict(Xte), index=yte.index)
    # Residual vs fitted
    fig = plt.figure(figsize=(4, 3))
    sns.scatterplot(x=pd.Series(chosen.predict(Xte)), y=resid, s=10, alpha=0.6)
    plt.axhline(0, color="#94a3b8", linestyle="--")
    plt.xlabel("Fitted")
    plt.ylabel("Residuals")
    plt.tight_layout()
    p = plots_dir / "residuals_vs_fitted.png"
    fig.savefig(p)
    plt.close(fig)
    explain["residuals_vs_fitted"] = f"{base_url}/residuals_vs_fitted.png"
    # QQ plot
    fig = sm.ProbPlot(resid).qqplot(line="s")
    plt.tight_layout()
    fig.savefig(plots_dir / "qq_plot.png")
    explain["qq_plot"] = f"{base_url}/qq_plot.png"
    # BP test
    try:
        import pandas as pd

        exog = sm.add_constant(pd.Series(chosen.predict(Xte)))
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
        if bp_p < 0.05:
            explain.setdefault("warnings", []).append(
                "Heteroscedasticity detected (Breusch-Pagan p<0.05)"
            )
    except Exception:
        pass


def add_shap_beeswarm(
    chosen,
    Xtr,
    Xte,
    plots_dir: Path,
    base_url: str,
    explain: Dict[str, Any],
    max_rows: int = 500,
):
    import shap
    import matplotlib.pyplot as plt

    sv = shap.Explainer(chosen, Xtr)(Xte[:max_rows])
    shap.plots.beeswarm(sv, show=False)
    fig = plt.gcf()
    p = plots_dir / "shap_beeswarm.png"
    fig.savefig(p)
    explain["shap_beeswarm"] = f"{base_url}/shap_beeswarm.png"
