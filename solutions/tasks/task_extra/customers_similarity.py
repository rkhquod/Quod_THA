import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from solutions.utils import load_data


def create_customer_product_matrix(df: pd.DataFrame):
    """
    Creates a matrix (DataFrame) where the rows are customers, the columns are products, and the values are the number
    of purchases of a given product by a given customer.
    """
    # Count the number of occurrences (purchases) of a given product by a given customer
    customer_product_counts = df.groupby(['customer_id', 'product_id']).size().reset_index(name='purchase_count')

    # Pivot - in rows: customer_id, in columns: product_id
    matrix = customer_product_counts.pivot_table(index='customer_id',
                                                 columns='product_id',
                                                 values='purchase_count',
                                                 fill_value=0)
    return matrix


def get_top_n_similar_users(matrix: pd.DataFrame, target_customer_id: str, n=10):
    """
    Returns a list (or DataFrame) of the n most similar users for a given customer_id, along with a similarity value.
    """

    if target_customer_id not in matrix.index:
        return pd.DataFrame(columns=['customer_id', 'similarity'])

    # Define vectors of users
    user_vectors = matrix.values  # shape (num_customers, num_products)

    # Define target indes (of user)
    target_index = matrix.index.get_loc(target_customer_id)

    # Calculate the cosine similarity of the entire matrix with respect to the selected row
    similarities = cosine_similarity(
        user_vectors[target_index].reshape(1, -1),  # target vector (1, num_products)
        user_vectors  # matrix (num_customers, num_products)
    )[0]  # Returns a 1D array of similarity values

    # Create DataFrame from similarities
    sim_df = pd.DataFrame({
        'customer_id': matrix.index,
        'similarity': similarities
    })

    # Remove customer itself from the list
    sim_df = sim_df[sim_df['customer_id'] != target_customer_id]

    # Sort descending by similarity and take the top n
    sim_df = sim_df.sort_values('similarity', ascending=False).head(n)
    return sim_df


def main():
    st.title("Customer comparison based on sales")

    df = load_data()

    # Create a user x product matrix
    matrix = create_customer_product_matrix(df)

    st.subheader("select customer ID to compare")
    customer_ids = matrix.index.tolist()
    chosen_customer_id = st.selectbox("Select customer_id", customer_ids)

    if chosen_customer_id:
        top_10 = get_top_n_similar_users(matrix, chosen_customer_id, n=10)

        st.write(f"**10 Most Similar Users** to Customer `{chosen_customer_id}`:")
        st.dataframe(top_10)

        st.bar_chart(
            data=top_10.set_index("customer_id")["similarity"],
            use_container_width=True
        )

        st.write(
            "Below you can compare the purchase distribution (number of transactions per product) for a selected user and, for example, one of the similar ones:")

        # Select from top_10 for comparison of schedules
        user_to_compare = st.selectbox("Select user to compare", top_10["customer_id"])

        if user_to_compare:
            # Get row for selected customer_id
            target_vector = matrix.loc[chosen_customer_id]
            compare_vector = matrix.loc[user_to_compare]

            # Make dataframe with two columns
            comparison_df = pd.DataFrame({
                "product_id": matrix.columns,
                f"{chosen_customer_id}": target_vector.values,
                f"{user_to_compare}": compare_vector.values
            })

            # Optionally, sort (e.g. descending) by the purchases of the selected customer
            comparison_df = comparison_df.sort_values(by=f"{chosen_customer_id}", ascending=False)

            st.write("Comparison of purchase quantities per product:")
            st.dataframe(comparison_df.head(20))

            comparison_df_top10 = comparison_df.head(10).set_index("product_id")
            st.bar_chart(comparison_df_top10)


if __name__ == "__main__":
    main()
