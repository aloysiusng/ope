import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from delivery import DeliveryQAgent,run_n_episodes2,DeliveryEnvironment
from sklearn.metrics.pairwise import cosine_similarity

def get_even_clusters(X, cluster_size):
    if len(X) <= cluster_size:
        # return an array of 0
        return np.zeros(len(X))
    n_clusters = int(np.ceil(len(X)/cluster_size))
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
    return clusters

def recommend(optimized_routes_df):
    transaction = pd.read_csv("obd_transaction_clean.csv")
    transaction.drop(["Invoice Date", "Base Qty", "Net Amount", "Discount Amount", "Unit Price", "CustomerCode", "GroupNameLevel2" ], axis = 1, inplace = True)
    pivot_df = pd.pivot_table(transaction,index = 'Invoice Number',columns = 'ProductCode',values = 'GroupNameLevel1',aggfunc = 'count')
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df.drop('Invoice Number', axis=1)
    co_matrix = pivot_df.T.dot(pivot_df)
    np.fill_diagonal(co_matrix.values, 0)
    cos_score_df = pd.DataFrame(cosine_similarity(co_matrix))
    cos_score_df.index = co_matrix.index
    cos_score_df.columns = np.array(co_matrix.index)

    optimized_routes_df['product_reco_1'] = None
    optimized_routes_df['product_reco_2'] = None
    optimized_routes_df['product_reco_3'] = None
    optimized_routes_df['product_reco_4'] = None
    optimized_routes_df['product_reco_5'] = None

    for i in range(len(optimized_routes_df)):
        product_code = optimized_routes_df['ProductCode'][i]
        reco_1, reco_2, reco_3, reco_4, reco_5 = cos_score_df[product_code].sort_values(ascending=False).index[1:6].tolist()
        optimized_routes_df.at[i,'product_reco_1'] = reco_1
        optimized_routes_df.at[i,'product_reco_2'] = reco_2
        optimized_routes_df.at[i,'product_reco_3'] = reco_3
        optimized_routes_df.at[i,'product_reco_4'] = reco_4
        optimized_routes_df.at[i,'product_reco_5'] = reco_5
    return optimized_routes_df

def generate_clusters(customer_df, transaction_df):


    # customer_df = pd.read_csv('obd_customer_clean.csv').drop('Unnamed: 0', axis=1).drop_duplicates(subset='CustomerCode', keep='first')
    # transaction_df = pd.read_csv('obd_transaction_clean.csv').drop_duplicates(subset=['CustomerCode', 'ProductCode'], keep='first')
    df = pd.merge(customer_df, transaction_df, on='CustomerCode')

    all_product_code = set(df['ProductCode'].unique())
    # one hot encode all ProductCode to df
    for product_code in all_product_code:
        df[product_code] = np.where(df['ProductCode'] == product_code, 1, 0)

    all_columns_list = transaction_df.columns.to_list() + customer_df.columns.to_list()

    # group by CustomerCode and sum all ProductCode
    while 'CustomerCode' in all_columns_list:
        all_columns_list.remove('CustomerCode')
    product_code_sum_df = df.drop(all_columns_list, axis=1).groupby('CustomerCode').sum().reset_index()

    # map the ProductCode values to df
    df.drop(all_product_code, axis=1, inplace=True)
    df = pd.merge(df, product_code_sum_df, on='CustomerCode')
    df = df.drop_duplicates(subset='CustomerCode', keep='first')

    # final df where all df will append to
    final_df = pd.DataFrame(columns=['CustomerCode', 'Latitude', 'Longitude', 'cluster', 'ProductCode', 'sequence_id', 'SalesmanCode', 'GroupNameLevel1', 'GroupNameLevel2', 'ProductNameTH'])
    for salesman_code in df['SalesmanCode'].unique():
        salesman_code_df = df[df['SalesmanCode'] == salesman_code]
        if len(salesman_code_df) == 1:
            continue

        # drop all columns except one hot encoded ProductCode
        df_for_clustering = salesman_code_df.drop(all_columns_list+['CustomerCode'], axis=1)

        # get even clusters of 45 custormers code in each cluster
        y_kmeans = get_even_clusters(df_for_clustering, 45)
        df_for_clustering['cluster'] = y_kmeans
        salesman_code_df['cluster'] = y_kmeans

        # heatmap of the clusters
        polar=df_for_clustering.groupby("cluster").mean().reset_index()
        polar=pd.melt(polar,id_vars=["cluster"])
        clust_means_wt = pd.DataFrame(df_for_clustering.groupby('cluster').mean(), columns=df_for_clustering.columns)
        # drop columns where the mean is 0
        clust_means_wt.drop(clust_means_wt.columns[clust_means_wt.mean() == 0], axis=1, inplace=True)

        # create a dictionary where the key is the cluster and the value is the ProductCode with the highest value in the cluster
        clust_means_wt_dict = {}
        for i in range(0, len(clust_means_wt)):
            clust_means_wt_dict[i] = clust_means_wt.iloc[i].idxmax()
        
        for cluster_num in clust_means_wt_dict.keys():
            x_arr = []
            y_arr = []
            test_locations = {}
            cluster_df = salesman_code_df[salesman_code_df['cluster'] == cluster_num][['CustomerCode', 'Latitude', 'Longitude', 'cluster', 'SalesmanCode', 'GroupNameLevel1', 'GroupNameLevel2', 'ProductNameTH']]
            cluster_df['ProductCode'] = clust_means_wt_dict[cluster_num]
            for i, row in cluster_df.iterrows():
                lat, lng = float(row['Latitude']), float(row['Longitude'])
                test_locations[row['CustomerCode']] = (lat, lng)
                x_arr.append(lat)
                y_arr.append(lng)
            x_arr = np.array(x_arr)
            y_arr = np.array(y_arr)

            env = DeliveryEnvironment(n_stops = len(x_arr),method = "distance", generate=(x_arr, y_arr))
            agent = DeliveryQAgent(env.observation_space,env.action_space)
            run_n_episodes2(env,agent)

            coordinates = []
            for i in env.stops:
                coordinates.append((env.x[i], env.y[i]))

            customer_code_to_index = {key:coordinates.index(value) for key, value in test_locations.items()}
            # map CustomerCode to sequence id from CustomerCode to index dict in pipeline
            cluster_df['sequence_id'] = cluster_df['CustomerCode'].map(customer_code_to_index)
            cluster_df['cluster'] = cluster_df['cluster'].apply(lambda x: f'{salesman_code}_{x}')
            final_df = pd.concat([final_df, cluster_df], ignore_index=True)
            print(f"Saved {salesman_code}_cluster_{cluster_num}")

    final_df = recommend(final_df)
    final_df.to_csv('../ETL/CSV/optimized_routes.csv')


