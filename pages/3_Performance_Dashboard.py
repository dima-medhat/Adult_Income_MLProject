import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

import matplotlib.pyplot as plt


st.title("📊 Performance Dashboard")

st.write(
    """
    Review the evaluation results of the model trained during the current
    Streamlit session.
    """
)

# sidebar 

with st.sidebar:

    st.title("💼 Adult Income ML")

    st.caption(
        """
        **Binary Classification**

        Built with Streamlit, Scikit-learn,
        XGBoost and SMOTE.
        """
    )

    st.divider()

    st.subheader("🗺️ Project Workflow")

    st.markdown(
        """
        ✅ Data Overview

        ⚙️ Model Training

        📊 Performance Dashboard

        🎯 Income Prediction
        """
    )

    st.divider()

    st.subheader("💡 Tip")

    st.info(
    """
    📊 **Page Overview**

    Evaluate the trained model using classification metrics, confusion matrix,
    ROC and Precision-Recall curves, and feature importance to better understand
    its predictive performance.
    """)

    st.divider()

    st.caption(
        """
        **Version 1.0**

        Built with ❤️ using
        Streamlit & Scikit-learn
        """
    )

# Check if a model has been trained

required_items = [
    "trained_model",
    "trained_results",
    "X_test",
    "y_test"
]

missing_items = [
    item
    for item in required_items
    if item not in st.session_state
]


if missing_items:
    st.warning(
        "No trained model is available yet. "
        "Open Model Training, train a model, then return to this page."
    )
    st.stop()


model = st.session_state["trained_model"]
results = st.session_state["trained_results"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]



# Predictions

y_pred = model.predict(X_test)


# Model summary

st.subheader("Model Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

summary_col1.metric(
    "Model",
    results["Model"]
)

summary_col2.metric(
    "Imbalance Method",
    results["Method"]
)

summary_col3.metric(
    "Test Samples",
    len(y_test)
)


st.divider()


# Evaluation metrics

st.subheader("Evaluation Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "Accuracy",
    f"{results['Accuracy']:.4f}"
)

metric_col2.metric(
    "Precision",
    f"{results['Precision']:.4f}"
)

metric_col3.metric(
    "Recall",
    f"{results['Recall']:.4f}"
)


metric_col4, metric_col5, metric_col6 = st.columns(3)

metric_col4.metric(
    "F1 Score",
    f"{results['F1 Score']:.4f}"
)

metric_col5.metric(
    "ROC-AUC",
    f"{results['ROC-AUC']:.4f}"
)

metric_col6.metric(
    "PR-AUC",
    f"{results['PR-AUC']:.4f}"
)


st.divider()


# -------------------------------------------------
# Metrics chart

st.subheader("Metric Comparison")

metric_chart_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC"
    ],
    "Score": [
        results["Accuracy"],
        results["Precision"],
        results["Recall"],
        results["F1 Score"],
        results["ROC-AUC"],
        results["PR-AUC"]
    ]
})

st.bar_chart(
    metric_chart_df,
    x="Metric",
    y="Score"
)


st.divider()


# -------------------------------------------------
# Confusion matrix and classification report
# -------------------------------------------------
st.subheader("Detailed Evaluation")

matrix_col, report_col = st.columns(2)


with matrix_col:

    st.markdown("### Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["<=50K", ">50K"],
        ax=ax
    )

    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    plt.close(fig)


with report_col:

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["<=50K", ">50K"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose().round(4)

    st.dataframe(
        report_df,
        use_container_width=True
    )


st.divider()


# -------------------------------------------------
# ROC and Precision-Recall curves
# -------------------------------------------------
st.subheader("Evaluation Curves")

roc_col, pr_col = st.columns(2)


with roc_col:

    st.markdown("### ROC Curve")

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title("ROC Curve")

    st.pyplot(fig)

    plt.close(fig)


with pr_col:

    st.markdown("### Precision-Recall Curve")

    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title("Precision-Recall Curve")

    st.pyplot(fig)

    plt.close(fig)


st.divider()

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------
st.divider()

st.subheader("Feature Importance")

supported_models = [
    "Decision Tree",
    "Random Forest",
    "XGBoost"
]

model_name = results["Model"]


if model_name in supported_models:

    try:
        # Access preprocessing and trained model steps
        preprocessor = model.named_steps["preprocessor"]
        trained_algorithm = model.named_steps["model"]

        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        # Get importance values
        importance_values = trained_algorithm.feature_importances_

        # Create feature-importance table
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values
        })

        # Clean feature names
        feature_importance_df["Feature"] = (
            feature_importance_df["Feature"]
            .str.replace("numerical__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        # Keep the 10 most important features
        top_features = (
            feature_importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(10)
            .sort_values(
                by="Importance",
                ascending=True
            )
        )

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_title(
            f"Top 10 Important Features — {model_name}"
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        #  table
        with st.expander("View feature importance table"):

            st.dataframe(
                feature_importance_df
                .sort_values(
                    by="Importance",
                    ascending=False
                )
                .head(10),
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.warning(
            f"Feature importance could not be displayed: {error}"
        )

else:

    st.info(
        "Feature importance is available for Decision Tree, "
        "Random Forest and XGBoost."
    )

st.success(
    "Evaluation completed. "
    "Open Income Prediction from the sidebar to make a prediction."
)