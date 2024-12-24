"""
Brings everything together:
 - Load & clean data
 - Perform exploratory analysis & plots
 - Train & compare multiple models
 - Save each trained model
 - Evaluate each model on the entire dataset
"""

import os
from data_preprocessing import load_data, clean_data
from visualizations import (
    plot_transactions_by_customer,
    plot_product_freq_for_2018,
    plot_top_5_products_last_6_months,
    plot_monthly_aggregate
)
from model_train import train_and_compare_models, save_models
from model_evaluate import evaluate_model

def main():
   
    DATA_PATH = "transactions_1.csv" 
    EXAMPLE_PRODUCT = "Volkswagen"   

    # 1) Load & clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # 2) Exploratory / Visualizations
    print("DataFrame shape:", df.shape)
    print(df.head(), "\n")

    plot_transactions_by_customer(df)
    plot_product_freq_for_2018(df, product_id=EXAMPLE_PRODUCT)
    plot_top_5_products_last_6_months(df)
    plot_monthly_aggregate(df)

    # 3) Train & Compare Multiple Models
    models_dict, metrics_dict = train_and_compare_models(df)

    # 4) Save Each Model
    save_models(models_dict)

    # 5) Evaluate each model on the entire dataset
    for model_name in models_dict.keys():
        model_file = f"{model_name}.pkl"
        evaluate_model(df, model_file)


if __name__ == "__main__":
    main()
