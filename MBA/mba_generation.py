import pandas as pd
import numpy as np
import json
from datetime import datetime
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# start timer ======================================================
print("===start timer===")
start = datetime.now()
# Read in the data ======================================================
with open("mba_config.json") as f:
    config = json.load(f)


df = pd.read_csv(config["transaction_file_name"])

print("===data cleaning===")
# Data processing ======================================================
df['Revenue'] = df.apply(lambda x: x['UnitPrice'] * x['BaseQty'] - x['DiscAmount'], axis=1)
basket_df = df.groupby(['InvoiceNumber', 'ProductCode'])['BaseQty'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNumber')

def remove_zero(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_clean_df = basket_df.applymap(remove_zero)
basket_clean_df = basket_clean_df[(basket_clean_df > 0).sum(axis=1) >= 10]
basket_clean_df.drop([col for col, val in basket_clean_df.mean().iteritems() if val < 0.05], axis=1, inplace=True)

print("===MBA===")
# MBA ============================================================================================================
frequent_itemsets = apriori(basket_clean_df, min_support=0.03,
                            use_colnames=True).sort_values(by='support', ascending=False).reset_index(drop=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

rules = association_rules(frequent_itemsets, metric='lift',
                        min_threshold=1).sort_values('lift', ascending=False).reset_index(drop=True)

print("=== cleaning rules===")
# Cleaning of rules ============================================================================================================
rules = rules[rules['confidence'] >= config["min_confidence"]]
rules = rules[rules['lift'] >= config["min_lift"]]

def updateAntRules(rules, dictUsed, colNameOut):
    for row in rules.itertuples():
        basket = []
        for pdt in list(row.antecedents):
            pdt_name = dictUsed[str(pdt)]
            basket.append(pdt_name)
        rules.at[row.Index, colNameOut] = ', '.join(basket)
    return rules

def updateConRules(rules, dictUsed, colNameOut):
    for row in rules.itertuples():
        basket = []
        for pdt in list(row.consequents):
            pdt_name = dictUsed[str(pdt)]
            basket.append(pdt_name)
        rules.at[row.Index, colNameOut] = ', '.join(basket)
    return rules

# Create mapping between Product Code and Product Name TH ======================================================
#  Group Level 1 and Group Level 2 ======================================================
code_to_name_dict = pd.Series(df["ProductNameTH"].values, index=df["ProductCode"]).to_dict()
code_to_group_level1_dict = pd.Series(df["GroupNameLevel1"].values, index=df["ProductCode"]).to_dict()
code_to_group_level2_dict = pd.Series(df["GroupNameLevel2"].values, index=df["ProductCode"]).to_dict()
code_to_name_dict = {str(key): str(value) for key, value in code_to_name_dict.items()}
code_to_group_level1_dict = {str(key): str(value) for key, value in code_to_group_level1_dict.items()}
code_to_group_level2_dict = {str(key): str(value) for key, value in code_to_group_level2_dict.items()}
total_revenue_pdt_code_dict = df.groupby(['ProductCode']).Revenue.sum().to_dict()
total_revenue_pdt_code_dict = {str(key): str(round(value, 2)) for key, value in total_revenue_pdt_code_dict.items()}

def understandMBA(rules):
    rules = updateAntRules(rules, code_to_name_dict, "ant_thai")
    rules = updateConRules(rules, code_to_name_dict, "con_thai")

    rules = updateAntRules(rules, code_to_group_level1_dict, "ant_grouplevel1")
    rules = updateConRules(rules, code_to_group_level1_dict, "con_grouplevel1")

    rules = updateAntRules(rules, code_to_group_level2_dict, "ant_grouplevel2")
    rules = updateConRules(rules, code_to_group_level2_dict, "con_grouplevel2")

    rules = updateAntRules(rules, total_revenue_pdt_code_dict, "ant_pdt_ttl_net_amt")
    rules = updateConRules(rules, total_revenue_pdt_code_dict, "con_pdt_ttl_net_amt")
    return rules

rules = understandMBA(rules)

#  Preparing clean data for Tableau (from frozenset to string) ===============================================================================================
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join([str(i) for i in list(x)])).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join([str(i) for i in list(x)])).astype("unicode")
rules.reset_index(inplace=True, drop=True)



#  split antecedents and consequents to multiple columns with labels e.g. antecedents 0, antecedents 1 ======================================================
# find coluumn number by name
print("=== Splitting columns ===")
def findColNumByName(df, colName):
    colNum = 0
    for i in df.columns:
        if i == colName:
            return colNum
        colNum += 1
    return -1
#  split antecedents and consequents to multiple columns with labels e.g. antecedents 0, antecedents 1 ======================================================
# split a column of string to multiple columns
def splitCol(df, colName):
    return df.join(df[colName].str.split(',', expand=True).rename(columns={0:colName+"0", 1:colName+"1", 2:colName+"2", 3:colName+"3", 4:colName+"4", 5:colName+"5", 6:colName+"6"}))

cols_to_split = ["antecedents", "ant_thai", "ant_grouplevel1", "ant_grouplevel2", "ant_pdt_ttl_net_amt", "consequents", "con_thai", "con_grouplevel1", "con_grouplevel2", "con_pdt_ttl_net_amt"]
for i in cols_to_split:
    rules[i] = rules[i].astype(str)
    rules = splitCol(rules, i)


# Output ============================================================================================================
# rules.to_csv("ETL_DATA_mba_rules_with_annotations.csv") 
rules.to_csv(config["output_file_name"]) 

end = datetime.now()
print( "Total time taken: ", end - start )