import pandas as pd
import streamlit as st


st.title("🎯 Income Prediction")

st.write(
    """
    Enter a person's information and use the model trained during the
    current Streamlit session to predict whether their annual income is
    greater than 50K.
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
    🎯 **Page Overview**

    Enter an individual's information and use the trained machine learning model
    to predict whether their annual income is greater than 50K or less than or
    equal to 50K, together with the prediction probabilities.
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



# -------------------------------------------------
# Check whether a model is available
# -------------------------------------------------
if "trained_model" not in st.session_state:
    st.warning(
        "No trained model is available yet. "
        "Open Model Training, train a model, then return to this page."
    )
    st.stop()


model = st.session_state["trained_model"]

trained_results = st.session_state.get(
    "trained_results",
    {}
)

cleaned_df = st.session_state.get(
    "cleaned_df"
)


if cleaned_df is None:
    st.error(
        "The prepared dataset is not available. "
        "Open Data Overview first."
    )
    st.stop()


# -------------------------------------------------
# Model information
# -------------------------------------------------
model_col1, model_col2 = st.columns(2)

model_col1.metric(
    "Active Model",
    trained_results.get(
        "Model",
        "Trained Model"
    )
)

model_col2.metric(
    "Imbalance Method",
    trained_results.get(
        "Method",
        "Baseline"
    )
)


st.divider()


# -------------------------------------------------
# Input form
# -------------------------------------------------
st.subheader("Personal Information")

with st.form("prediction_form"):

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:

        age = st.number_input(
            "Age",
            min_value=17,
            max_value=100,
            value=35
        )

        workclass = st.selectbox(
            "Workclass",
            sorted(
                cleaned_df["workclass"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        education = st.selectbox(
            "Education",
            sorted(
                cleaned_df["education"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        educational_num = st.number_input(
            "Educational Number",
            min_value=1,
            max_value=20,
            value=10
        )

        marital_status = st.selectbox(
            "Marital Status",
            sorted(
                cleaned_df["marital-status"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    with input_col2:

        occupation = st.selectbox(
            "Occupation",
            sorted(
                cleaned_df["occupation"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        relationship = st.selectbox(
            "Relationship",
            sorted(
                cleaned_df["relationship"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        race = st.selectbox(
            "Race",
            sorted(
                cleaned_df["race"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        gender = st.selectbox(
            "Gender",
            sorted(
                cleaned_df["gender"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        capital_gain = st.number_input(
            "Capital Gain",
            min_value=0,
            value=0,
            step=100
        )


    with input_col3:

        capital_loss = st.number_input(
            "Capital Loss",
            min_value=0,
            value=0,
            step=100
        )

        hours_per_week = st.number_input(
            "Hours per Week",
            min_value=1,
            max_value=100,
            value=40
        )

        native_country = st.selectbox(
            "Native Country",
            sorted(
                cleaned_df["native-country"]
                .dropna()
                .astype(str)
                .unique()
            )
        )


    submit_prediction = st.form_submit_button(
        "Predict Income",
        type="primary",
        use_container_width=True
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------
if submit_prediction:

    if hours_per_week <= 34:
        work_hours_category = "Part-time"

    elif hours_per_week <= 44:
        work_hours_category = "Full-time"

    elif hours_per_week <= 60:
        work_hours_category = "Overtime"

    else:
        work_hours_category = "Extreme Overtime"


    input_data = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "education": [education],
        "educational-num": [educational_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country],
        "work-hours-category": [work_hours_category]
    })


    try:

        prediction = model.predict(
            input_data
        )[0]


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_data
            )[0]

            low_income_probability = probabilities[0]
            high_income_probability = probabilities[1]

        else:

            low_income_probability = None
            high_income_probability = None


        st.divider()

        st.subheader("Prediction Result")


        if prediction == 1:

            st.success(
                "Predicted income class: **Greater than 50K**"
            )

        else:

            st.info(
                "Predicted income class: **Less than or equal to 50K**"
            )


        if high_income_probability is not None:

            probability_col1, probability_col2 = (
                st.columns(2)
            )

            probability_col1.metric(
                "Probability of <=50K",
                f"{low_income_probability:.2%}"
            )

            probability_col2.metric(
                "Probability of >50K",
                f"{high_income_probability:.2%}"
            )

            st.progress(
                float(high_income_probability),
                text=(
                    "Model confidence for income greater than 50K"
                )
            )


        with st.expander("View entered information"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )