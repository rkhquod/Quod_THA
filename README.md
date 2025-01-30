# Sales Forecasting and Customer Behavior Analysis

This project aims to analyze customer behavior and predict future transactions using machine learning models. The dataset includes customer transactions with features such as `Customer ID`, `Product`, and `Time stamp`. The goal is to build a flexible and accurate model to forecast transactions and provide insights into customer and product dynamics.

---

## **Project Structure Overview**

The project is organized as follows:

```plaintext
├── artifacts/                  # Model artifacts
├── configs/                    # Configuration files
├── data/                       # Raw and processed datasets
├── src/                        # Core project code
│   ├── models/                 # Model implementations
│   ├── training/               # Training logic
│   ├── evaluation/             # Evaluation metrics
│   ├── preprocessing/          # Data preprocessing logic
│   ├── visualization/          # Plotting utilities
├── output/                     # Saved visualizations
├── run.py                      # Entry point for the pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---

## **Key Features**

### **Data Visualization**

- Ordered plot of total transactions per customer (from most to least active) saved in `outputs/product_count_per_customer.png`.
- Monthly transaction frequency for Peugeot from 2017 to 2020, saved in  `outputs/monthly_Peugeot_transactions.png`
- Top products with the highest sales over the last six months, with seasonality analysis. This is plotted for each month until the last date. The plot is saved in  `outputs/top_products.png`

### **Model Building**

- Predict the total number of transactions for the next three months per customer in 2019.
- Supported models: Ridge Model, Random Forest, XGBoost, and Neural Network.
- Model configuration is indicated in  `configs/model_config.yaml`

### **Performance Evaluation**

- Metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE) are calculated for each model.
- Artifacts for each model including hyperparameters, model, training and evaluation logs are saved in the `artifacts/model_name` folder.

---

## **How to Run**

1. **Install Dependencies**:

```bash
   pip install -r requirements.txt
 ```

2. **Run the pipeline**:

```bash
   python run.py [steps]
 ```

Available steps:

- **preprocess**: Preprocess the raw data.

- **visualize**: Generate visualizations.

- **train**: Train the selected model.

- **evaluate**: Evaluate the model's performance.

- **all**: Run all steps sequentially.


**Example:**

```bash
python run.py preprocess train evaluate
```

To execute the training process, we choose a model by entering the corresponding number:

- 1 : NeuralNet
- 2 : XGBoost
- 3 : RandomForest
- 4 : RidgeML

3. **Performance Metrics**:


The performance of each model was evaluated using Mean Absolute Error (MAE) and Mean Squared Error (MSE). The task was to predict the sum of sales for each customer from February to April 2019. For comparison, a benchmark model was included, which predicts sales based on the sum of sales from the last 3-month period (November 2018 to January 2019).

| Model          | MAE   | MSE   |
|----------------|-------|-------|
| Benchmark      | 31.11 | 9033  |
| Ridge          | 21.24 | 5429  |
| Neural Network | 17.94 | 3954  |
| XGBoost        | 17.13 | 3334  |
| Random Forest  | 17.2  | 2914  |
