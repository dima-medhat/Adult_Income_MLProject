import pandas as pd
import streamlit as st

from utils.preprocessing import clean_and_engineer_data


st.title("📂 Data Overview")

st.write(
    """
    Explore the Adult Income dataset used in this classification project.
    You may upload the Adult Income CSV to replace the included dataset.
    """
)
with st.sidebar:

    st.title("💼 Adult Income ML")

    st.caption(
        """
        **Binary Classification**
        """)

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
    📖 **Page Overview**

    This page provides an overview of the Adult Income dataset,
    including its structure, feature types, missing values,
    and target class distribution before model training.
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

# Load the included Adult Income dataset by default
try:
    raw_df = pd.read_csv("data/adult.csv")

except FileNotFoundError:
    st.error(
        "The included dataset was not found. "
        "Add adult.csv inside the data folder."
    )
    st.stop()


# Optional upload
uploaded_file = st.file_uploader(
    "Upload Adult Income CSV (optional)",
    type=["csv"]
)


# Replace the default dataset only when a file is uploaded
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded successfully.")

    except Exception as error:
        st.error(f"Unable to read the uploaded file: {error}")
        st.stop()


# Clean and prepare whichever dataset is being used
try:
    cleaned_df = clean_and_engineer_data(raw_df)

except Exception as error:
    st.error(f"Unable to prepare the dataset: {error}")
    st.stop()


# Save data for the other pages
st.session_state["raw_df"] = raw_df
st.session_state["cleaned_df"] = cleaned_df


# Dataset summary
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rows",
    cleaned_df.shape[0]
)

col2.metric(
    "Columns",
    cleaned_df.shape[1]
)

col3.metric(
    "Features",
    cleaned_df.shape[1] - 1
)

col4.metric(
    "Target Classes",
    cleaned_df["income"].nunique()
)


st.divider()


# Dataset preview
st.subheader("Dataset Preview")

st.dataframe(
    cleaned_df.head(10),
    use_container_width=True
)


st.divider()


# Income class distribution
st.subheader("Income Class Distribution")

income_counts = (
    cleaned_df["income"]
    .map({
        0: "<=50K",
        1: ">50K"
    })
    .value_counts()
)

chart_col, table_col = st.columns([2, 1])

with chart_col:
    st.bar_chart(income_counts)

with table_col:
    distribution_df = pd.DataFrame({
        "Income Class": income_counts.index,
        "Count": income_counts.values,
        "Percentage": (
            income_counts.values
            / income_counts.sum()
            * 100
        ).round(2)
    })

    st.dataframe(
        distribution_df,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# Feature information
st.subheader("Feature Information")

feature_info = pd.DataFrame({
    "Feature": cleaned_df.columns,
    "Data Type": cleaned_df.dtypes.astype(str).values,
    "Unique Values": cleaned_df.nunique().values
})

st.dataframe(
    feature_info,
    use_container_width=True,
    hide_index=True
)


st.divider()


# Numerical summary
st.subheader("Numerical Summary")

st.dataframe(
    cleaned_df.describe().round(2),
    use_container_width=True
)


st.info(
    "The dataset is ready. Open Model Training from the sidebar."
)