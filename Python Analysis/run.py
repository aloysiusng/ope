from datetime import datetime
import pandas as pd
from recommender import generate_recommender
from mba import generate_mba
from clustering import generate_clusters

print("starting batch process......")
start = datetime.now()

print("loading transaction data......")
transaction_file_name = "../ETL/CSV/obd_transaction.csv"
df = pd.read_csv(transaction_file_name, index_col=0)

print("transaction data loaded")
print(df)

# generate_mba(df, 2, 0.5, 0.05) # df, min_lift, min_confidence, min_support
generate_recommender(df)

# print("loading customer data......")
# customer_file_name = "../ETL/CSV/obd_customer_unique.csv"
# df_customer = pd.read_csv(customer_file_name, index_col=0)
# print(df_customer)

# print("customer data loaded")
# print("starting route optimization......")
# transaction_df = df.drop_duplicates(subset=['CustomerCode', 'ProductCode'], keep='first')
# generate_clusters(df_customer, transaction_df)

end = datetime.now()
print( "Total time taken for batch run: ", end - start )