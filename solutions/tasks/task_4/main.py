import altair as alt
import datetime
import pandas as pd
import streamlit as st

from solutions.const import DATE_COLNAME, PRODUCT_ID_COLNAME
from solutions.utils import load_data


def main():
    st.set_page_config(layout="wide")
    st.title("Top Products - Last 6 Months")

    # Load data
    df_data = load_data()

    # Set default date
    default_date = datetime.date(2019, 1, 31)

    # Widget to choose data
    chosen_date = st.date_input(
        label="Choose the reference date",
        value=default_date
    )

    # Widget to choose top N products
    n_top_products = st.number_input(
        label="Number of top products to retrieve",
        min_value=1,
        max_value=50,
        value=5
    )

    # Widget to select the number of recent months
    last_n_months = st.number_input(
        label="How many months to consider",
        min_value=1,
        max_value=24,
        value=6
    )

    # Convert selected date to Timestamp
    chosen_timestamp = pd.Timestamp(chosen_date).tz_localize("UTC")

    # Calculate the starting date
    start_date = chosen_timestamp - pd.DateOffset(months=last_n_months)

    # Filter data to a selected range
    df_last_period = df_data[
        (df_data[DATE_COLNAME] >= start_date) &
        (df_data[DATE_COLNAME] <= chosen_timestamp)
        ]

    # Group by product
    product_sales = (
        df_last_period
        .groupby(PRODUCT_ID_COLNAME)
        .size()
        .reset_index(name="transaction_count")
    )

    # Sort and selecting TOP N
    top_products = (
        product_sales
        .sort_values(by="transaction_count", ascending=False)
        .head(n_top_products)
    )

    # Display short information
    st.write(
        f"**Top {n_top_products} products** between "
        f"{start_date.date()} and {chosen_timestamp.date()} "
        f"(last {last_n_months} months)"
    )

    # Mmake plot
    chart = (
        alt.Chart(top_products)
        .mark_bar()
        .encode(
            x=alt.X(f"{PRODUCT_ID_COLNAME}:N", title="Product ID", sort=None),
            y=alt.Y("transaction_count:Q", title="Transaction Count"),
            tooltip=[f"{PRODUCT_ID_COLNAME}:N", "transaction_count:Q"]
        )
        .properties(
            width=700,
            height=400,
            title=f"Top {n_top_products} Products in Last {last_n_months} Months"
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # Make table
    st.write("Detail table of top products:")
    st.dataframe(top_products.reset_index(drop=True))


if __name__ == "__main__":
    main()
