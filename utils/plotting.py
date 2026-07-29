import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


def plot_confusion_matrix(model, X_test, y_test):
    """
    Create a confusion matrix figure for a trained model.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["<=50K", ">50K"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Confusion Matrix")

    fig.tight_layout()

    return fig


def plot_roc_curve(model, X_test, y_test):
    """
    Create the ROC curve for a trained model.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title("ROC Curve")

    fig.tight_layout()

    return fig


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Create the Precision-Recall curve for a trained model.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title("Precision-Recall Curve")

    fig.tight_layout()

    return fig


def plot_metric_comparison(results_df, metric_name):
    """
    Compare one evaluation metric across all models.

    results_df must contain:
    - Model
    - the selected metric column
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        results_df["Model"],
        results_df[metric_name]
    )

    ax.set_title(f"{metric_name} Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="x",
        rotation=20
    )

    fig.tight_layout()

    return fig