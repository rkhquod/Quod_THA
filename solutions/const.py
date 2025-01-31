import os

# directories
DATASET_DIR = 'dataset'
RESULTS_DIR = os.path.join('solutions', 'tasks', 'results')


CUSTOMER_ID_COLNAME = 'customer_id'
DATE_COLNAME = 'date'
PRODUCT_ID_COLNAME = 'product_id'

data_1_filename = 'transactions_1.csv'
data_2_filename = 'transactions_2.csv'

data_1_filepath = os.path.join(DATASET_DIR, data_1_filename)
data_2_filepath = os.path.join(DATASET_DIR, data_2_filename)
