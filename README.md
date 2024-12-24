# Sales Forecasting and Analysis

This project provides a mini data science workflow to analyze and predict customer behavior and product dynamics.  
It includes:
- Loading and cleaning data,
- Some exploratory visualizations,
- Training **at least two models** (here, **Linear Regression** and **Random Forest Regressor**),
- Comparing their performance and saving the trained models.

## Project Structure

```bash
.
├── data_preprocessing.py      # Data loading and cleaning
├── visualizations.py          # Visualization functions (exploration)
├── model_train.py             # Data preparation, training, and comparison of multiple models
├── model_evaluate.py          # Loading and evaluating a saved model
├── main.py                    # Main entry point orchestrating the workflow
├── transactions_1.csv
├── transactions_2.csv         # Example CSV files containing transactions
└── README.md                  # This documentation file


```
### File Roles

1. **`data_preprocessing.py`**  
   - Contains functions for loading and cleaning the dataset (CSV file):  
     - Converting the date to datetime format,  
     - Removing invalid rows or dates,  
     - Handling duplicates, etc.

2. **`visualizations.py`**  
   - Defines several plotting functions (using `matplotlib`) to:  
     - Display the number of transactions per customer,  
     - Analyze transaction frequency by month for a given product,  
     - View the top 5 products over the past 6 months,  
     - Identify potential monthly seasonality.

3. **`model_train.py`**  
   - Prepares the data for modeling (monthly grouping by customer),  
   - Trains **multiple models** (e.g., Linear Regression, Random Forest),  
   - Compares their metrics (MAE, RMSE) on the same test set,  
   - Allows **saving** each trained model to a `.pkl` file.

4. **`model_evaluate.py`**  
   - Loads one of the saved models,  
   - Evaluates its overall performance (MAE, RMSE),  
   - Optionally displays a scatter plot of "actual values vs. predictions."

5. **`main.py`**  
   - Serves as the main script. It:  
     1. Loads and cleans data from `transactions_x.csv`,  
     2. Generates key visualizations,  
     3. Trains multiple models and compares their performance,  
     4. Saves each model,  
     5. Evaluates each model on the entire dataset (or a subset).

6. **`transactions_x.csv`**  
   - Example transaction files.


## Installation & Execution

1. **Clone or copy** this repository to your local machine.

2. **Install dependencies** (if needed):
   ```bash
   pip install pandas numpy scikit-learn matplotlib

3. **Run the main script** :
    ```bash
    python main.py
    ```

   This will display the graphs and some information in the terminal (e.g., formatted DataFrame output).  
The script will also train multiple models (e.g., LinearRegression, RandomForestRegressor), compare their MAE/RMSE, and save each model (`.pkl`).
