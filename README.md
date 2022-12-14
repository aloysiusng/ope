# G1T4 IS455 OPE (Data Analytics in Asia)
---
# How to run
CD into Python Analysis folder and pip install -r requirements.txt

Double click on run.bat (takes around 20 to 30mins to finsih running the script)

---
# Folder structure
- ETL
  - CSV
     - obd_customer.csv
     - obd_customer_unique.csv
     - obd_transaction.csv
     - obd_visit_plan.csv
     - optimized_routes_reco.csv
  - PKL
     - obd_customer.pkl
     - obd_transaction.pkl
     - obd_visit_plan.pkl
  - bigquery.yml
  - ETL.ipynb
- OPE
  - EDA
    - EDA.ipynb
  - MBA
     - mba_config.json
     - mba_generation.py
     - mba_rules.csv
     - mba.sh
  - Python Analysis
     - clustering.py
     - delivery.py
     - mba.py
     - recommender.py
     - requirements.txt
     - run.bat
     - run.py
  - Visualisation
     - ~Osotspa Sales_272.twbr
     - ~Osotspa Sales_12672.twbr
     - Osotspa Market Basket Analysis.twb
     - Osotspa Sales.twb
---
## ETL Folder
Contains ETL code which we use to fetch data from Osotspa BigQuery. Output csv files is stored here.

ETL folder is outside of OPE folder (gitinit) as the csv files are too large to be pushed to GitHub

## EDA Folder
Contains EDA.ipynb which we used to explore the data

## MBA Folder
Contains our code for MBA and a mba.sh (batch script) to be run every night

## Python Analysis Folder
Contains our code for generating recommendation.csv and optimized_routes_reco.csv.

Recommendation.csv is generated from recommender.py which uses Item Based Collaborative Filtering (IBCF). Recommendation.csv is used in Tableau to show recommended products

Optimized_routes_reco.csv is generated from clustering.py which used k-Means Clustering to group similar stalls and Reinforcement Learning to generate shortest route. Optimized_routes_reco.csv is used in Tableau to show the shortest route to take for a van sales visit

Run.bat (batch script) to be run every night

## Visualisation Folder
Contains our Tableau dashboard
 


