# Sales Forecast and Analysis

## System Requirements

This project requires a **Linux machine** (e.g., Ubuntu) and **Python 3.12** installed.

## Setup and Installation

To set up the environment and install necessary dependencies, run the following commands:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

## Running the Solutions
* Each task is implemented as a separate module inside the solutions/tasks/ directory.
* Tasks `1`,`2` and `4` solutions are visualisations presented in `streamlit` framework.
* Task `4` solution consists of 3 modules:
  * `training` - train and save models
  * `inference` - predict on already trained models
  * `validate` - validate of trained models (in the `streamlit` app)
* You can run the scripts using the following commands:

Be sure, that You have good PYTHONPATH (base directory)
```bash
export PYTHONPATH=$(pwd)
```

### Task 1: Transactions per Customer
```bash
streamlit run solutions/tasks/task_1/main.py
```

### Task 2: Transaction Frequency per Month for a Given Product in 2018
```bash
streamlit run solutions/tasks/task_2/main.py
```

### Task 3: Customer Transactions Forecasting
#### Training the Model:
```bash
streamlit run solutions/tasks/task_3/training.py
```

Inference and validation are possible for default dates during training ("2019-01-31").

#### Making Predictions using the Saved Model:
```bash
python solutions/tasks/task_3/inference.py
````

#### Validate saved models:
```bash
streamlit run solutions/tasks/task_3/validate.py
````

### Task 4: Top 5 Products Over the Last 6 Months
```bash
streamlit run solutions/tasks/task_4/main.py
```

### Task Extra
There are 2 additional tasks with interesting visualization

```bash
streamlit run solutions/tasks/task_extra/customers_similarity.py
```

```bash
streamlit run solutions/tasks/task_extra/transactions_heatmap.py
```
