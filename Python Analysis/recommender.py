import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

def generate_recommender(df):

    print("Starting recommender......")
    start = datetime.now()

    print("Mapping df table......")
    pivot_df = pd.pivot_table(df[['InvoiceNumber','ProductCode','BaseQty']],index = 'InvoiceNumber', columns = 'ProductCode', values = 'BaseQty', aggfunc = 'sum', fill_value = 0)
    
    # each row is an order and each column is a product and the values are the counts of the products in each of the orders.
    pivot_df = pivot_df.reset_index().drop('InvoiceNumber', axis=1)
    print(pivot_df)

    # transform pivot table into a co-occurrence matrix by taking the dot product of the pivot table and its transpose
    print("Creating co-occurrence matrix......")
    co_matrix = pivot_df.T.dot(pivot_df)
    np.fill_diagonal(co_matrix.values, 0)
    print(co_matrix)

    # transform the co-occurrence matrix into a matrix of cosine similarities
    print("calculating cosine similarity score on total qty for every transaction......")
    cos_score_df = pd.DataFrame(cosine_similarity(co_matrix))
    cos_score_df.index = co_matrix.index
    cos_score_df.columns = np.array(co_matrix.index)
    print(cos_score_df)

    # construct a dataframe that contains the top 10 most similar products to the given product
    print("mapping top 10 products based on cosine similarity score.....")
    recommendation_df = pd.DataFrame(columns=['ProductInput', 'ProductCode', 'CosineSimilarity'])
    for item in cos_score_df:
        top_products = pd.DataFrame(cos_score_df[item].sort_values(ascending=False).iloc[1:11])
        top_products.reset_index(inplace=True)
        top_products['ProductCode'] = top_products['ProductCode'].astype(int)
        top_products['ProductInput'] = item
        top_products.rename(columns={item: 'CosineSimilarity'},inplace=True)
        recommendation_df = pd.concat([recommendation_df, top_products], ignore_index=True)

    # get product name TH, group level 1 name, group level 2 name for each product input
    print("mapping product name TH, group level 1 name, group level 2 name for each product input.....")

    # create dictionary table of df consisting of unique productnameTH, GroupNameLevel1, GroupNameLevel 2 based on product code
    product_dict = df[['ProductCode','ProductNameTH','GroupNameLevel1','GroupNameLevel2']].drop_duplicates().set_index('ProductCode').to_dict(orient='index')

    # map product name TH, group level 1 name, group level 2 name for each product input based on product input
    recommendation_df['PI_ProductNameTH'] = recommendation_df['ProductInput'].map(product_dict).map(lambda x: x['ProductNameTH'])
    recommendation_df['PI_GroupNameLevel1'] = recommendation_df['ProductInput'].map(product_dict).map(lambda x: x['GroupNameLevel1'])
    recommendation_df['PI_GroupNameLevel2'] = recommendation_df['ProductInput'].map(product_dict).map(lambda x: x['GroupNameLevel2'])

    recommendation_df.set_index('ProductInput', inplace=True)

    print(recommendation_df)

    print("exporting recommendation to CSV......")
    recommendation_df.to_csv('../ETL/CSV/recommendation.csv')

    end = datetime.now()

    print( "Total time taken for recommendation: ", end - start )