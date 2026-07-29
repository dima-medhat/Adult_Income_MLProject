import streamlit as st


st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="wide"
)

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
        Compare the same algorithm using
        **Unbalanced** and **SMOTE**
        to evaluate the effect of class
        imbalance handling.
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
    st.divider()


# Title and introduction

st.title("💼 Adult Income Prediction")

st.markdown(
    """
    ### Machine Learning Classification Project

    Predict whether a person's annual income is **greater than 50K**
    or **less than or equal to 50K** using five supervised machine-learning
    algorithms, preprocessing, imbalance handling, hyperparameter tuning,
    and model evaluation.
    """
)


st.divider()


# Available algorithms

st.subheader("Available Algorithms")

algorithm_col1, algorithm_col2, algorithm_col3, algorithm_col4, algorithm_col5 = (
    st.columns(5)
)

with algorithm_col1:
    with st.container(border=True):
        st.markdown("### 🧮")
        st.markdown("**Naive Bayes**")
        st.caption("Probabilistic")

with algorithm_col2:
    with st.container(border=True):
        st.markdown("### 🌳")
        st.markdown("**Decision Tree**")
        st.caption("Tree-Based")

with algorithm_col3:
    with st.container(border=True):
        st.markdown("### 🌲")
        st.markdown("**Random Forest**")
        st.caption("Ensemble")

with algorithm_col4:
    with st.container(border=True):
        st.markdown("### 📐")
        st.markdown("**Support Vector Machine**")
        st.caption("Margin-Based")

with algorithm_col5:
    with st.container(border=True):
        st.markdown("### 🚀")
        st.markdown("**XGBoost**")
        st.caption("Boosting")


st.divider()


# Application pages

st.subheader("Application Pages")

page_col1, page_col2 = st.columns(2)

with page_col1:
    st.info(
        """
        ### 1. Data Overview

        Upload and prepare the Adult Income dataset.
        Review its structure and target distribution.
        """
    )

    st.info(
        """
        ### 2. Model Training

        Select an algorithm, configure RandomizedSearchCV,
        and train the model.
        """
    )

with page_col2:
    st.info(
        """
        ### 3. Performance Dashboard

        Review model metrics, confusion matrix,
        classification report, and evaluation curves.
        """
    )

    st.info(
        """
        ### 4. Income Prediction

        Enter feature values and receive the predicted
        income class with probabilities.
        """
    )


st.divider()



# Getting started

st.success(
    "Use the sidebar to navigate through the application. "
    "Start with the Data Overview page and upload the original dataset."
)