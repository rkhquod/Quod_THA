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

- Ordered plot of total transactions per customer (from most to least active).
- Monthly transaction frequency for any product in 2018.
- Top products with the highest sales over the last six months, with seasonality analysis. This is plotted for each month until the last date.
- All these plots are **saved in outputs folder**

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

The performance of each model is evaluated using MAE and MSE. Below is a summary of the results:

| Model          | MAE   | MSE   |
|----------------|-------|-------|
| Ridge          | 21.82 | 5752  |
| XGBoost        | 18.89 | 4240  |
| Neural Network | 18.27 | 4105  |
| Random Forest  | 17.2  | 2914  |
