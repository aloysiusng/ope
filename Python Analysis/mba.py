import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from datetime import datetime

def generate_mba(df, min_lift=2, min_confidence=0.5, min_support=0.05):

    print("starting market basket analysis......")
    start = datetime.now()

    # transform transaction table
    print("transforming transaction data......")
    basket_df = df.groupby(['InvoiceNumber', 'ProductCode']).size().unstack(fill_value=0).astype(bool)
    print(basket_df)


    # create frequent item sets
    # rules = minimum support 5%
    print(f"creating frequent itemsets with minimum support of {min_support*100}%....")
    frequent_itemsets  = apriori(basket_df, min_support=0.05, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print(frequent_itemsets)

    # create association rules
    print("creating association rules [support, confidence, lift,  leverage, conviction] with minimum lift of 1 .......")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1).sort_values('lift', ascending=False)
    print(rules)

    # clean rules
    print(f"further cleaning of association rules [confidence >= {min_confidence}, lift >= {min_confidence}]....")
    rules = rules[rules['confidence'] >= min_confidence]
    rules = rules[rules['lift'] >= min_lift]

    #  Preparing clean data for Tableau (from frozenset to string) 
    print("formatting data....")
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join([str(i) for i in list(x)])).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join([str(i) for i in list(x)])).astype("unicode")
    
    # split antecedents and consequents into separate columns
    print("splitting antecedents and consequents into separate columns....")
    rules = rules.join(rules['antecedents'].str.split(',', expand=True).add_prefix('antecedent_'))

    # Explode the consequents column to get the consequents in separate rows
    print("Exploding consequents column....")
    rules = rules.assign(**{'consequents': rules['consequents'].str.split(',')}).explode('consequents')
    print(rules)

    # mapping rules to relevant information.... 
    print("mapping rules to relevant information....")
    code_to_name_dict = pd.Series(df["ProductNameTH"].values, index=df["ProductCode"]).to_dict()
    code_to_group_level1_dict = pd.Series(df["GroupNameLevel1"].values, index=df["ProductCode"]).to_dict()
    code_to_group_level2_dict = pd.Series(df["GroupNameLevel2"].values, index=df["ProductCode"]).to_dict()
    code_to_name_dict = {str(key): str(value) for key, value in code_to_name_dict.items()}
    code_to_group_level1_dict = {str(key): str(value) for key, value in code_to_group_level1_dict.items()}
    code_to_group_level2_dict = {str(key): str(value) for key, value in code_to_group_level2_dict.items()}
    total_revenue_pdt_code_dict = df.groupby(['ProductCode']).NetAmount.sum().to_dict()
    total_revenue_pdt_code_dict = {str(key): str(round(value, 2)) for key, value in total_revenue_pdt_code_dict.items()}

    
    def updateRules(rules, dictUsed, colNameOut, ant_con):
        for row in rules.itertuples():
            basket = []
            if ant_con == 1:
                for pdt in list(row.antecedents.split(',')):
                    pdt_name = dictUsed[str(pdt.strip())]
                    basket.append(pdt_name)
                rules.at[row.Index, colNameOut] = ', '.join(basket)
            else:
                pdt_name = dictUsed[str(row.consequents.strip())]
                print(pdt_name)
                rules.at[row.Index, colNameOut] = pdt_name

        return rules

    rules = updateRules(rules, code_to_name_dict, "ant_thai", 1)
    rules = updateRules(rules, code_to_name_dict, "con_thai", 2)

    rules = updateRules(rules, code_to_group_level1_dict, "ant_grouplevel1", 1)
    rules = updateRules(rules, code_to_group_level1_dict, "con_grouplevel1", 2)

    rules = updateRules(rules, code_to_group_level2_dict, "ant_grouplevel2", 1)
    rules = updateRules(rules, code_to_group_level2_dict, "con_grouplevel2", 2)

    rules = updateRules(rules, total_revenue_pdt_code_dict, "ant_pdt_ttl_net_amt", 1)
    rules = updateRules(rules, total_revenue_pdt_code_dict, "con_pdt_ttl_net_amt", 2)
    print(rules)

    rules.reset_index(inplace=True, drop=True)
    # find coluumn number by name
    print("Splitting columns ...")
    # split a column of string to multiple columns
    def splitCol(df, colName):
        return df.join(df[colName].str.split(',', expand=True).rename(columns={0:colName+"0", 1:colName+"1", 2:colName+"2", 3:colName+"3", 4:colName+"4", 5:colName+"5", 6:colName+"6"}))

    cols_to_split = ["antecedents", "ant_thai", "ant_grouplevel1", "ant_grouplevel2", "ant_pdt_ttl_net_amt", "consequents", "con_thai", "con_grouplevel1", "con_grouplevel2", "con_pdt_ttl_net_amt"]
    for i in cols_to_split:
        rules[i] = rules[i].astype(str)
        rules = splitCol(rules, i)

    # export rules to csv
    print("exporting association rules to CSV....")
    rules.to_csv("../ETL/CSV/association_rules.csv")

    end = datetime.now()
    print( "Total time taken for market basket analysis: ", end - start )
