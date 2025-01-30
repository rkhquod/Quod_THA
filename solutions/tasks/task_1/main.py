import altair as alt
import streamlit as st

from solutions.const import CUSTOMER_ID_COLNAME
from solutions.utils import load_data


def main():
    # set the layout to wide to have more space for the horizontal graph
    st.set_page_config(layout="wide")

    st.title("Transactions per Customer (Descending Order)")

    # Load data
    df_data = load_data()

    # Group transactions per customer
    transactions_per_customer = (
        df_data
        .groupby(CUSTOMER_ID_COLNAME)
        .size()
        .reset_index(name="transaction_count")
    )
    transactions_per_customer.sort_values(
        by="transaction_count", ascending=False, inplace=True
    )

    # Slider for selecting TOP N customers
    max_customers = len(transactions_per_customer)
    top_n = st.slider(
        "How many most active customers to display?",
        min_value=1,
        max_value=max_customers,
        value=min(50, max_customers)
    )

    # Cut to TOP N
    top_customers = transactions_per_customer.head(top_n)

    # Create an Altair Chart
    chart = (
        alt.Chart(top_customers)
        .mark_bar()
        .encode(
            x=alt.X(f"{CUSTOMER_ID_COLNAME}:N", sort=None, title="Customer ID"),
            y=alt.Y("transaction_count:Q", title="Transaction Count"),
            tooltip=[f"{CUSTOMER_ID_COLNAME}:N", "transaction_count:Q"]
        )
        .properties(width=1200, height=500)
    )

    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
