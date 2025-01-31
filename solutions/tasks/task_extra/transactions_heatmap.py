import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from solutions.utils import load_data


def main():
    st.title("Visualization of the number of transactions")

    df_data = load_data()

    df_data["day_of_week"] = df_data["date"].dt.dayofweek

    df_data['hour'] = df_data['date'].dt.hour

    # Group data, count the number of occurrences (transactions) depending on the day of the week and time
    dow_hour = df_data.groupby(['day_of_week', 'hour'])['customer_id'].count().reset_index()

    # create a pivot table with day_of_week as index and hour as columns
    dow_hour_pivot = dow_hour.pivot(index='day_of_week', columns='hour', values='customer_id')

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(dow_hour_pivot, cmap="YlGnBu", ax=ax)
    ax.set_title("Distribution of the number of transactions by day of the week and hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day of the week (0=Monday, 6=Sunday)")

    # Plot
    st.pyplot(fig)


if __name__ == "__main__":
    main()
