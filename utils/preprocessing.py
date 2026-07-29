import pandas as pd

def clean_and_engineer_data(df):

    # Work on a copy
    df_model = df.copy()

    # Remove duplicate rows
    df_model.drop_duplicates(inplace=True)

    # Replace ? with missing values
    df_model.replace("?", pd.NA, inplace=True)

    # Remove missing values
    df_model.dropna(inplace=True)

    # Clean spaces in text columns
    text_columns = df_model.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df_model[column] = (
            df_model[column]
            .astype("string")
            .str.strip()
        )

    # Drop fnlwgt
    if "fnlwgt" in df_model.columns:
        df_model.drop(columns=["fnlwgt"], inplace=True)

    # Convert target
    if "income" in df_model.columns:
        df_model["income"] = (
            df_model["income"]
            .str.replace(".", "", regex=False)
            .map({
                "<=50K": 0,
                ">50K": 1
            })
        )

    # Create work-hours category
    if "hours-per-week" in df_model.columns:

        bins = [0, 34, 44, 60, float("inf")]

        labels = [
            "Part-time",
            "Full-time",
            "Overtime",
            "Extreme Overtime"
        ]

        df_model["work-hours-category"] = pd.cut(
            df_model["hours-per-week"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

    # Group rare countries
    if "native-country" in df_model.columns:

        country_counts = df_model["native-country"].value_counts()

        top_countries = country_counts[
            country_counts >= 100
        ].index

        df_model["native-country"] = (
            df_model["native-country"].where(
                df_model["native-country"].isin(top_countries),
                "Other"
            )
        )

    return df_model