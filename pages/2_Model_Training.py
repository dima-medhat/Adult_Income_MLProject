import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from utils.preprocessing import clean_and_engineer_data


st.title("⚙️ Model Training")

st.write(
    """
    Select a machine-learning algorithm, configure its hyperparameters,
    and train it using the Adult Income dataset.
    """
)

# sidebar

with st.sidebar:

    st.title("💼 Adult Income ML")

    st.caption(
        """
        **Binary Classification**
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
    ⚙️ **Page Overview**

    Train and optimize a machine learning model by selecting an algorithm,
    configuring hyperparameters, and choosing an imbalance handling method.
    The trained model is then evaluated and prepared for prediction.
    """
    )

    st.divider()

    st.caption(
        """
        **Version 1.0**

        Built with ❤️ using
        Streamlit & Scikit-learn
        """
    )


# Load dataset

if "cleaned_df" in st.session_state:

    df_model = st.session_state["cleaned_df"].copy()

else:

    try:
        raw_df = pd.read_csv("data/adult.csv")
        df_model = clean_and_engineer_data(raw_df)

        st.info(
            "The included Adult Income dataset is being used."
        )

    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()



# Split features and target

X = df_model.drop(columns=["income"])
y = df_model["income"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Detect feature types

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


numerical_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# Select algorithm

algorithm_name = st.selectbox(
    "Select an algorithm",
    [
        "Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "XGBoost"
    ]
)

# select balanced or unbalanced
imbalance_method = st.selectbox(
    "Class Imbalance Handling",
    [
        "Unbalanced",
        "SMOTE"
    ],
    help=(
        "Unbalanced uses the original training data. "
        "SMOTE creates synthetic samples for the minority class."
    )
)




st.divider()

st.subheader("Hyperparameters")


# Algorithm-specific inputs

if algorithm_name == "Naive Bayes":

    var_smoothing = st.number_input(
        "Variance smoothing",
        min_value=1e-12,
        max_value=1e-5,
        value=1e-9,
        format="%.10f"
    )

    model = GaussianNB(
        var_smoothing=var_smoothing
    )

   


elif algorithm_name == "Decision Tree":

    max_depth = st.slider(
        "Maximum depth",
        min_value=2,
        max_value=30,
        value=10
    )

    min_samples_split = st.slider(
        "Minimum samples split",
        min_value=2,
        max_value=20,
        value=2
    )

    min_samples_leaf = st.slider(
        "Minimum samples leaf",
        min_value=1,
        max_value=10,
        value=1
    )

    criterion = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"]
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )


elif algorithm_name == "Random Forest":

    n_estimators = st.slider(
        "Number of trees",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    max_depth = st.slider(
        "Maximum depth",
        min_value=2,
        max_value=30,
        value=15
    )

    min_samples_split = st.slider(
        "Minimum samples split",
        min_value=2,
        max_value=20,
        value=2
    )

    min_samples_leaf = st.slider(
        "Minimum samples leaf",
        min_value=1,
        max_value=10,
        value=1
    )

    max_features = st.selectbox(
        "Maximum features",
        ["sqrt", "log2"]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )



elif algorithm_name == "SVM":

    c_value = st.select_slider(
        "C value",
        options=[0.1, 1.0, 10.0],
        value=1.0)

    gamma_value = st.selectbox(
        "Gamma",
        ["scale", "auto"])

  

    model = SVC(
        C=c_value,
        kernel="rbf",
        gamma=gamma_value,
        probability=True,
        random_state=42)


else:

    n_estimators = st.slider(
        "Number of boosting rounds",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.01, 0.03, 0.05, 0.1, 0.2],
        value=0.1
    )

    max_depth = st.slider(
        "Maximum depth",
        min_value=2,
        max_value=10,
        value=5)

    subsample = st.slider(
        "Subsample",
        min_value=0.5,
        max_value=1.0,
        value=1.0,
        step=0.1
    )

    colsample_bytree = st.slider(
        "Column sample by tree",
        min_value=0.5,
        max_value=1.0,
        value=1.0,
        step=0.1
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

 


st.divider()


# Train model

if st.button(
    "Train Model",
    type="primary",
    use_container_width=True
):

    if imbalance_method == "SMOTE":

        training_pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model)])
        method_name = "SMOTE"

    else:

        training_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)])
        method_name = "Unbalanced"


    with st.spinner(
        f"Training {algorithm_name}. Please wait..."
    ):

        try:

            training_pipeline.fit(
                X_train,
                y_train
            )

            y_pred = training_pipeline.predict(
                X_test
            )

            if hasattr(
                training_pipeline,
                "predict_proba"
            ):

                y_score = training_pipeline.predict_proba(
                    X_test
                )[:, 1]

            else:

                y_score = training_pipeline.decision_function(
                    X_test
                )


            results = {
                "Model": algorithm_name,
                "Method": method_name ,
                "Accuracy": accuracy_score(
                    y_test,
                    y_pred
                ),
                "Precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                "ROC-AUC": roc_auc_score(
                    y_test,
                    y_score
                ),
                "PR-AUC": average_precision_score(
                    y_test,
                    y_score)}


            st.session_state["trained_model"] = (
                training_pipeline)

            st.session_state["trained_results"] = (
                results)

            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test


            st.success(
                f"{algorithm_name} trained successfully.")


            metric_col1, metric_col2, metric_col3 = (
                st.columns(3))

            metric_col1.metric(
                "Accuracy",
                f"{results['Accuracy']:.4f}")

            metric_col2.metric(
                "Precision",
                f"{results['Precision']:.4f}")

            metric_col3.metric(
                "Recall",
                f"{results['Recall']:.4f}")


            metric_col4, metric_col5, metric_col6 = (
                st.columns(3))

            metric_col4.metric(
                "F1 Score",
                f"{results['F1 Score']:.4f}")

            metric_col5.metric(
                "ROC-AUC",
                f"{results['ROC-AUC']:.4f}")

            metric_col6.metric(
                "PR-AUC",
                f"{results['PR-AUC']:.4f}")


            st.info(
                "The trained model is saved in the current "
                "Streamlit session and can be used by the "
                "Performance Dashboard and Income Prediction pages.")

        except Exception as error:

            st.error(
                f"Training failed: {error}")