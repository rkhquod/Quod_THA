import streamlit as st
import altair as alt

from solutions.const import DATE_COLNAME, PRODUCT_ID_COLNAME
from solutions.utils import load_data

def main():
    # Set the layout to "water" for a larger work area
    st.set_page_config(layout="wide")

    st.title("Transaction Frequency per Month for a Given Product (2018)")

    # Load data
    df_data = load_data()

    # Filtering data to 2018
    df_2018 = df_data[df_data[DATE_COLNAME].dt.year == 2018]

    if df_2018.empty:
        st.warning("No data for 2018.")
        return

    # List of unique product_ids in 2018 data
    product_ids_2018 = df_2018[PRODUCT_ID_COLNAME].unique().tolist()

    if not product_ids_2018:
        st.warning("No unique products in 2018.")
        return

    # Select a product from the list
    selected_product = st.selectbox(
        "Select product_id:",
        product_ids_2018
    )

    # Filtering data by selected product_id
    product_data = df_2018[df_2018["product_id"] == selected_product].copy()

    if product_data.empty:
        st.warning(f"No transactions in 2018 for the product '{selected_product}'.")
        return

    # Create a year_month column (or month directly) for grouping
    product_data["year_month"] = product_data[DATE_COLNAME].dt.to_period("M")
    monthly_transactions = (
        product_data
        .groupby("year_month")
        .size()
        .reset_index(name="transaction_count")
    )

    # Convert year_month to string (e.g. '2018-01', '2018-02', ...)
    monthly_transactions["year_month_str"] = monthly_transactions["year_month"].astype(str)

    # Create a chart in Altair
    chart = (
        alt.Chart(monthly_transactions)
        .mark_line(point=True)
        .encode(
            x=alt.X("year_month_str:N", title="Month (YYYY-MM)"),
            y=alt.Y("transaction_count:Q", title="Transaction Count"),
            tooltip=["year_month_str:N", "transaction_count:Q"]
        )
        .properties(
            width=800,
            height=400,
            title=f"Monthly Transactions in 2018 for Product: {selected_product}"
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # Show raw data
    st.dataframe(monthly_transactions)


if __name__ == "__main__":
    main()
