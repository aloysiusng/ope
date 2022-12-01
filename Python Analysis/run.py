from datetime import datetime
import pandas as pd
from recommender import generate_recommender
from mba import generate_mba

print("starting batch process......")
start = datetime.now()

print("loading transaction data......")
transaction_file_name = "../ETL/CSV/obd_transaction.csv"
df = pd.read_csv(transaction_file_name, index_col=0)
print("transaction data loaded")
print(df)

# generate_mba(df)
generate_recommender(df)

end = datetime.now()
print( "Total time taken for batch run: ", end - start )