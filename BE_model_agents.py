# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # B.E. Model Agents
# **By Steven, Orion, Jason, and Michael**

# + {"toc": true, "cell_type": "markdown"}
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-Dictionary" data-toc-modified-id="Data-Dictionary-1">Data Dictionary</a></span></li><li><span><a href="#Environment" data-toc-modified-id="Environment-2">Environment</a></span></li><li><span><a href="#Acquisition" data-toc-modified-id="Acquisition-3">Acquisition</a></span></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-4">Preparation</a></span><ul class="toc-item"><li><span><a href="#Rename-and-Drop-Columns" data-toc-modified-id="Rename-and-Drop-Columns-4.1">Rename and Drop Columns</a></span><ul class="toc-item"><li><span><a href="#Lowercase-all-column-names" data-toc-modified-id="Lowercase-all-column-names-4.1.1">Lowercase all column names</a></span></li><li><span><a href="#Drop-Columns" data-toc-modified-id="Drop-Columns-4.1.2">Drop Columns</a></span></li><li><span><a href="#Rename-Columns" data-toc-modified-id="Rename-Columns-4.1.3">Rename Columns</a></span></li></ul></li><li><span><a href="#Rename-Variable-Values" data-toc-modified-id="Rename-Variable-Values-4.2">Rename Variable Values</a></span><ul class="toc-item"><li><span><a href="#Transaction-Side" data-toc-modified-id="Transaction-Side-4.2.1">Transaction Side</a></span></li></ul></li><li><span><a href="#Filter" data-toc-modified-id="Filter-4.3">Filter</a></span><ul class="toc-item"><li><span><a href="#Select-Finished-Transactions" data-toc-modified-id="Select-Finished-Transactions-4.3.1">Select Finished Transactions</a></span></li><li><span><a href="#Drop-Outliers" data-toc-modified-id="Drop-Outliers-4.3.2">Drop Outliers</a></span></li></ul></li><li><span><a href="#Encode-Columns" data-toc-modified-id="Encode-Columns-4.4">Encode Columns</a></span><ul class="toc-item"><li><span><a href="#Various-ID-fields" data-toc-modified-id="Various-ID-fields-4.4.1">Various ID fields</a></span></li></ul></li><li><span><a href="#Dates" data-toc-modified-id="Dates-4.5">Dates</a></span><ul class="toc-item"><li><span><a href="#Correcting-dates" data-toc-modified-id="Correcting-dates-4.5.1">Correcting dates</a></span></li><li><span><a href="#Convert-dates-to-datetime-type" data-toc-modified-id="Convert-dates-to-datetime-type-4.5.2">Convert dates to datetime type</a></span></li><li><span><a href="#Add-columns-for-year-and-quarter-of-sale" data-toc-modified-id="Add-columns-for-year-and-quarter-of-sale-4.5.3">Add columns for year and quarter of sale</a></span></li></ul></li><li><span><a href="#Location" data-toc-modified-id="Location-4.6">Location</a></span></li><li><span><a href="#Brokerage-34's-Texas-Transactions-Filter" data-toc-modified-id="Brokerage-34's-Texas-Transactions-Filter-4.7">Brokerage 34's Texas Transactions Filter</a></span></li><li><span><a href="#Add-in-ZIP-codes-and-Lat/Long" data-toc-modified-id="Add-in-ZIP-codes-and-Lat/Long-4.8">Add in ZIP codes and Lat/Long</a></span></li><li><span><a href="#Exclude-Jan-2018-data" data-toc-modified-id="Exclude-Jan-2018-data-4.9">Exclude Jan 2018 data</a></span></li><li><span><a href="#Remove-low-dollar-transactions" data-toc-modified-id="Remove-low-dollar-transactions-4.10">Remove low-dollar transactions</a></span></li><li><span><a href="#Impute" data-toc-modified-id="Impute-4.11">Impute</a></span><ul class="toc-item"><li><span><a href="#Impute-the-gross-commission-from-the-average-commission-rate" data-toc-modified-id="Impute-the-gross-commission-from-the-average-commission-rate-4.11.1">Impute the gross commission from the average commission rate</a></span></li><li><span><a href="#Impute-commissions-for-agents-and-brokerages" data-toc-modified-id="Impute-commissions-for-agents-and-brokerages-4.11.2">Impute commissions for agents and brokerages</a></span></li></ul></li><li><span><a href="#Create-Derivative-Dataframes" data-toc-modified-id="Create-Derivative-Dataframes-4.12">Create Derivative Dataframes</a></span><ul class="toc-item"><li><span><a href="#2018-Dataframe" data-toc-modified-id="2018-Dataframe-4.12.1">2018 Dataframe</a></span></li><li><span><a href="#Agent-Dataframe" data-toc-modified-id="Agent-Dataframe-4.12.2">Agent Dataframe</a></span></li></ul></li><li><span><a href="#Data-Sanity/Validation-Checks" data-toc-modified-id="Data-Sanity/Validation-Checks-4.13">Data Sanity/Validation Checks</a></span></li></ul></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-5">Exploration</a></span><ul class="toc-item"><li><span><a href="#Summary-Stats" data-toc-modified-id="Summary-Stats-5.1">Summary Stats</a></span></li><li><span><a href="#By-Agent" data-toc-modified-id="By-Agent-5.2">By Agent</a></span><ul class="toc-item"><li><span><a href="#By-Quintile" data-toc-modified-id="By-Quintile-5.2.1">By Quintile</a></span></li></ul></li><li><span><a href="#Sale-Price/Commission" data-toc-modified-id="Sale-Price/Commission-5.3">Sale Price/Commission</a></span></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-5.4">Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Commission-Strategy" data-toc-modified-id="Commission-Strategy-5.4.1">Commission Strategy</a></span></li><li><span><a href="#Add-number-of-ZIP-codes-each-agent-sells-in" data-toc-modified-id="Add-number-of-ZIP-codes-each-agent-sells-in-5.4.2">Add number of ZIP codes each agent sells in</a></span></li><li><span><a href="#Year-over-year-difference-in-total-sales" data-toc-modified-id="Year-over-year-difference-in-total-sales-5.4.3">Year over year difference in total sales</a></span></li><li><span><a href="#Calculate-High-Performance-Threshold-for-Feb---May-2019" data-toc-modified-id="Calculate-High-Performance-Threshold-for-Feb---May-2019-5.4.4">Calculate High Performance Threshold for Feb - May 2019</a></span></li><li><span><a href="#Unique-ZIP-Codes-per-Agent" data-toc-modified-id="Unique-ZIP-Codes-per-Agent-5.4.5">Unique ZIP Codes per Agent</a></span></li><li><span><a href="#Agent-cut-v.-Brokerage-cut" data-toc-modified-id="Agent-cut-v.-Brokerage-cut-5.4.6">Agent cut v. Brokerage cut</a></span></li><li><span><a href="#Number-of-Top-ZIP-Codes-in-Agent's-Market" data-toc-modified-id="Number-of-Top-ZIP-Codes-in-Agent's-Market-5.4.7">Number of Top ZIP Codes in Agent's Market</a></span></li></ul></li><li><span><a href="#Proceeds" data-toc-modified-id="Proceeds-5.5">Proceeds</a></span></li><li><span><a href="#Statistical-Tests" data-toc-modified-id="Statistical-Tests-5.6">Statistical Tests</a></span></li></ul></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-6">Modeling</a></span><ul class="toc-item"><li><span><a href="#Encoding-Columns" data-toc-modified-id="Encoding-Columns-6.1">Encoding Columns</a></span></li><li><span><a href="#Feature-Selection" data-toc-modified-id="Feature-Selection-6.2">Feature Selection</a></span></li><li><span><a href="#Train-Test-Split" data-toc-modified-id="Train-Test-Split-6.3">Train Test Split</a></span></li><li><span><a href="#Scale" data-toc-modified-id="Scale-6.4">Scale</a></span></li><li><span><a href="#Cross-Validation" data-toc-modified-id="Cross-Validation-6.5">Cross Validation</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-6.5.1">Logistic Regression</a></span></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-6.5.2">Decision Tree</a></span></li><li><span><a href="#Visualize-Model" data-toc-modified-id="Visualize-Model-6.5.3">Visualize Model</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-6.5.4">Random Forest</a></span></li><li><span><a href="#KNN" data-toc-modified-id="KNN-6.5.5">KNN</a></span></li></ul></li><li><span><a href="#Grid-Search" data-toc-modified-id="Grid-Search-6.6">Grid Search</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-6.6.1">Logistic Regression</a></span></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-6.6.2">Decision Tree</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-6.6.3">Random Forest</a></span></li></ul></li><li><span><a href="#Final-Model" data-toc-modified-id="Final-Model-6.7">Final Model</a></span><ul class="toc-item"><li><span><a href="#Pickle-Model-for-Web-App" data-toc-modified-id="Pickle-Model-for-Web-App-6.7.1">Pickle Model for Web App</a></span></li><li><span><a href="#Test-Pickled-Model" data-toc-modified-id="Test-Pickled-Model-6.7.2">Test Pickled Model</a></span></li></ul></li></ul></li></ul></div>
# -

# ## Data Dictionary

# - **agent_id** -> unique identifier for each agent
# - **agent_name** -> bogus name for anonymity reasons
# - **commission_anniversary** -> when the time comes to renegotiate the agent's split rate
# - **account_id** -> unique identifier of parent company
# - **brokerage_id** -> unique identifier for a subunit of a parent company (i.e., each brokerage_id is a subunit of account_id). A brokerage is a subunit of an agency.
# - **brokerage_name** -> bogus name for anonymity reasons
# - **commission_schedule_id** -> unique identifier for a commission schedule
# - **commission_schedule_effective_start_at** -> when the commission schedule starts
# - **commission_schedule_effective_end_at** -> when the commission schedule ends
# - **commission_schedule_active** -> whether the commmission schedule is active or not
# - **commission_schedule_strategy/com_plan** -> how the commission is calculated
#     - Mapping of encoder:
#         - 0: Accumulation Strategy
#         - 1: Flat Rate Strategy
#         - 2: Rolling Accumulation Strategy
# - **transaction_id/trans_id** -> a unique identifier for the transaction
# - **transaction_number** -> BT 'year' 'month' 'day' 'transaction_count'
# 	- unique to a single account, use transaction_id
# - **transaction_contracted_at** -> when the buyers and sellers signed contract to begin transaction
# - **transaction_closed_at** -> when the transaction was closed
# - **transaction_effective_at/sale_date** -> an override for when the transaction actually closed (there might be some last minute changes)
# - **transaction_status/sale_status** -> Open, ~DONT WORRY ABOUT THESE~ ,Cda Sent, Complete, or Fell through
# - **transaction_sales_amount/sale_amount** ->
# - **transaction_list_amount** -> set by users (WILL TRY TO PULL FROM LISTING SIDE SYSTEM)
# - **property_information** -> address of the property
# - **earned_side_count/com_split** -> a strange representation of how much credit an agent gets for a transaction
#     - can be split between agents on a side
# - **earned_volume** -> typically is the same as the sales amount
# 	- can be split between agents on a side
# - **tags/property_use** -> usage of the property (i.e., residential)
# - **transaction_side/trans_side** ->
# 	- Listing Side -> The agent is representing the seller of the property
# 	- Selling Side -> The agent is representing the buyer of the property
# - **transaction_price_override/price_override** ->
# - **standard_commission_type** -> the regular payout from a transaction
# - **standard_commission_gci_amount/com_gross** ->
# 	- the 3% || base value the brokerage took as commission
# 	- before splitting with agent
# - **standard_commission_agent_net_amount/com_agent_net** -> how much the agent took home
# - **standard_commission_brokerage_net_amount/com_brokerage_net** -> how much the brokerage took
# - **total_fees_charged_against_brokerage** -> the sum total of all the fees charged to the brokerage
# - **total_fees_charged_against_agent** -> the sum total of all fees charged to the agent; these can be paid by the brokerage
# - **total_fees_paid_on_transaction** -> $total\_fees\_charged\_against\_agent + total\_fees\_charged\_against\_brokerage$
# - **total_liabilities_against_brokerage** -> the amount the brokerage is liable to pay out to other parties.
# 	- These parties include the franchise, marketing, vendors, brokerage concessions and many more...
# - **total_liabilities_against_agent** -> the amount the agent is liable to pay to other parties.
# - **total_liabilities_on_transaction** -> $total\_liabilities\_against\_brokerage + total\_liabilities\_against\_the\_agent$
# - **total_brokerage_income_collected_from_agent_fees** -> $total\_fees\_charged\_against\_agent - total\_liabilities\_against\_the\_agent$
# - **final_brokerage_income_after_all_liabilities_are_paid_out** -> $brokerage\_net\_amount + total\_brokerage\_income\_collected\_from\_agent\_fees - total\_liabilities\_on\_transaction$
# - **bonus_commission_type** -> any extra money paid out
# - **bonus_commission_agent_net_amount** ->
# - **bonus_commission_brokerage_net_amount** ->
# - **listing_transaction_guid** -> used for joining with listing data

# ## Environment 

# +
from pprint import pprint
import operator
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.stats as stats
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
import seaborn as sns

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# -

# ## Acquisition

df_raw = pd.read_csv('agents_with_transactions.csv')
print(f"Observations: {df_raw.shape[0]}")
print(f"Variables: {df_raw.shape[1]}")

# Preserve raw data

df = df_raw.copy()

# ## Preparation

# ### Rename and Drop Columns

# #### Lowercase all column names

df.rename(columns=lambda col: col.lower(), inplace=True)

# #### Drop Columns

# +
# Agent ID provides a unique identifier already.
df.drop(columns='agent_name', inplace=True)

# Most of the data is missing and it's not clear what value this data provides.
df.drop(columns='commission_anniversary', inplace=True)

# Brokerage ID provides a unique identifier.
df.drop(columns='brokerage_name', inplace=True)

# It's not clear what value this data provides; there's 3554 unique schedule id's. Many agents each have more than one commission schedule id.
df.drop(columns='commission_schedule_id', inplace=True)

# This columns speak to when agents' commission plan starts and ends and is not necessary for our project.
df.drop(columns='commission_schedule_effective_start_at', inplace=True)
df.drop(columns='commission_schedule_effective_end_at', inplace=True)

# Some rows say TRUE when they should be FALSE and this data is not useful since we already know what was received in commissions.
df.drop(columns='commission_schedule_active', inplace=True)

# This column provides the same information as transaction_id and transaction_effective_at.
df.drop(columns='transaction_number', inplace=True)

# The column transaction_effective_at provides better information.
df.drop(columns='transaction_closed_at', inplace=True)

# This column has many NaNs.
df.drop(columns='transaction_list_amount', inplace=True)

# Unreliable data; we don't know if the value reflects reality.
df.drop(columns='earned_side_count', inplace=True)

# This columns is almost always the product of earned_sales_amount and transaction_sales_amount.
df.drop(columns='earned_volume', inplace=True)

# Need more information about the many variations of this.
df.drop(columns="tags", inplace=True)

# It is either one value or null and is not useful.
df.drop(columns='standard_commission_type', inplace=True)

# This information is about commission adjustments. We have decided to keep this after all.
# df.drop(columns='transaction_price_override', inplace=True)

# Post commission split and too many unknowns going into these values.
df.drop(columns=['total_fees_charged_against_brokerage',
                  'total_fees_charged_against_agent',
                  'total_fees_paid_on_transaction',
                  'total_liabilities_against_brokerage',
                  'total_liabilities_against_agent',
                  'total_liabilities_on_transaction',
                  'total_brokerage_income_collected_from_agent_fees',
                  'final_brokerage_income_after_all_liabilities_are_paid_out',], inplace=True)

# Mostly NaNs.
df.drop(columns=['bonus_commission_type',
                 'bonus_commission_agent_net_amount',
                 'bonus_commission_brokerage_net_amount',], inplace=True)

# Drop the transaction GUID; we are not using this now.
df.drop(columns="listing_transaction_guid", inplace=True)
# -

# #### Rename Columns
# To shorter, more descriptive names.

# +
new_col_names = {'commission_schedule_strategy': 'com_plan', 
                 'transaction_id': 'trans_id',
                 'transaction_contracted_at': 'contract_date',
                 'transaction_effective_at': 'sale_date', 
                 'transaction_status': 'sale_status', 
                 'transaction_sales_amount': 'sale_amount',
                 'property_information': "property_address",
                 'transaction_side': 'trans_side',
                 'transaction_price_override': 'price_override',
                 'standard_commission_gci_amount': 'com_gross', 
                 'standard_commission_agent_net_amount': 'com_agent_net', 
                 'standard_commission_brokerage_net_amount': 'com_brokerage_net'
                }

df = df.rename(columns=new_col_names)
# -

# ### Rename Variable Values

# #### Transaction Side
#
# Selling/Buying is clearer than Listing/Selling

df.trans_side.replace("LISTING_SIDE", "Selling", inplace=True)
df.trans_side.replace("SELLING_SIDE", "Buying", inplace=True)

# ### Filter

# #### Select Finished Transactions

# There are too many issues with handling "OPEN" or other types of transactions. E.g., is the sale amount accurate? Let's just proceed with the ones we know are finished and the money has come in.

before = len(df)
df = df[df.sale_status == 'COMPLETE']
print(f'Dropped {before - len(df)} rows.')

# #### Drop Outliers

# Drop row where sales_amount is > $1 billion because at least one of the transactions was more than this amount and it was an entry error. We are keeping this code in case it creeps back in.

before = len(df)
df = df[df.sale_amount < 1_000_000_000]
print(f'Dropped {before - len(df)} rows.')

# ### Encode Columns

# #### Various ID fields
#
# The raw values for the \_id columns are long randomized strings whose uniqueness is the only thing of value. Let's make them more manageable by encoding.

# +
to_encode = ['agent_id',
             'brokerage_id',
             'trans_id',
             'account_id']

label_encoders = dict()  # save the encoder objects
for col in to_encode:
    le = LabelEncoder().fit(df[col])
    label_encoders[col] = le
    df[col] = le.transform(df[col])
# -

# ### Dates

# #### Correcting dates
# These dates were corrected because they had bad years. The corrected year was inferred from the dates for transaction_closed_at and transaction_contracted_at columns.

df.loc[df.trans_id == 8017, 'sale_date'] = "2018-08-31 12:00:00 UTC"
df.loc[df.trans_id == 5675, 'sale_date'] = "2019-07-22 12:00:00 UTC"
df.loc[df.trans_id == 3423, 'sale_date'] = "2019-11-12 12:00:00 UTC"
df.loc[df.trans_id == 5141, 'sale_date'] = "2019-06-14 12:00:00 UTC"
df.loc[df.trans_id == 6246, 'sale_date'] = "2018-08-31 12:00:00 UTC"

# #### Convert dates to datetime type
# Drop the time as well

df['sale_date'] = pd.to_datetime(df.sale_date, errors = 'coerce')
df.sale_date = df.sale_date.dt.date
df['sale_date'] = pd.to_datetime(df.sale_date, errors = 'coerce')  # this is necessary to convert the column back to datetime type

# #### Add columns for year and quarter of sale

df = df.assign(sale_year=df.sale_date.dt.year)
df = df.assign(sale_quarter=df.sale_date.dt.quarter)
df = df.assign(sale_month=df.sale_date.dt.month)

# ### Location

# Separate property address information into separate columns

working = df.property_address.str.extract(r'(.+?)\s+-\s+(.+?)\s*-\s*(.+?)\s*-\s*(\d+)?')
working.rename(columns={0:'address', 1: 'city', 2: 'state', 3: 'zip'}, inplace=True)

# Drop the rows that have no address information

before = len(working)
working = working.dropna(subset=['address'])
print(f'Dropped {before - len(working)} rows.')

# Join the new address columns

df = pd.merge(left=df, right=working, left_on=df.index, right_on=working.index).drop(columns='key_0')

# Use abbreviation of state instead of full name

# +
df['state'] = df['state'].str.upper()

states_to_replace = {
    'TEXAS': 'TX',
    'CALIFORNIA': 'CA',
    'KANSAS': 'KS',
    'KENTUCKY': 'KY',
    'GEORGIA': 'GA',
    'INDIANA': 'IN',
    'ILLINOIS': 'IL',
    'COLORADO': 'CO',
    'FLORIDA': 'FL',
    'NORTH CAROLINA': 'NC',
    'HAWAII': 'HI',
    'IOWA': 'IA',
    'NEW YORK': 'NY',
    'OREGON': 'OR',
    'KANSA': 'KS',
    'VIRGINIA': 'VA',
    'NORTH CAROLINE': 'NC',
    'ARIZONA': 'AZ',
    'MICHIGAN': 'MI',
    'ALABAMA': 'AL',
    'WASHINGTON': 'WA',
    'NHY': 'NY',
    'CONNETICUT': 'CT',
    'NEBRASKA': 'NE',
    'KANSA': 'KS',
    'TC': 'TCCC',
    'BC': 'BCCC',
    '84': '8444',
    'NA': 'NAAA'
}

df['state'].replace(states_to_replace, inplace=True)

before = len(df)
df = df.drop(df[df['state'].map(len) != 2].index)
print(f'Dropped {before - len(df)} rows.')

df[['address', 'city', 'state', 'zip']].to_csv('50_TX_addresses.csv', sep=',', index=False)
# -

# ### Brokerage 34's Texas Transactions Filter

# First make a backup of df

df_all = df.copy()

# Extract only those Texas transactions connected with Brokerage 34, the target brokerage.

before = len(df)
df = df[(df.brokerage_id == 34) & (df.state == "TX")]
print(f'Dropped {before - len(df)} rows.')

# How many agents from Brokerage 34 do we start with?

df.agent_id.nunique()

# ### Add in ZIP codes and Lat/Long

df2 = pd.read_csv('lat_lng.csv')
df = pd.merge(left=df, right=df2, on=df.index).drop(columns='key_0')

# ### Exclude Jan 2018 data
# It's incomplete.

before = len(df)
df = df[df.sale_date >= "2018-02-01"]
after = len(df)
print(f"Dropped {before - after} rows.")

# ### Remove low-dollar transactions

before = len(df)
df = df[(df.sale_amount > 20_000)]
print(f'Dropped {before - len(df)} rows.')

# ### Impute

# #### Impute the gross commission from the average commission rate

com_gross_avg = df.com_gross.sum() / df.sale_amount.sum()
print(f'Filling {df.com_gross.isna().sum()} rows.')
df.com_gross.fillna(df.sale_amount * com_gross_avg, inplace=True)

# #### Impute commissions for agents and brokerages

# What is the agent's average proportion of take from the sale amount?

print(f'Filled {df.com_agent_net.isna().sum()} rows.')
com_agent_avg = df.com_agent_net.sum() / df.sale_amount.sum()
df.com_agent_net.fillna(com_agent_avg * df.sale_amount, inplace=True)

# What is the brokerage's average proportion of take from the sale amount?

print(f'Filled {df.com_brokerage_net.isna().sum()} rows.')
com_brokerage_avg = df.com_brokerage_net.sum() / df.sale_amount.sum()
df.com_brokerage_net.fillna(com_brokerage_avg * df.sale_amount, inplace=True)

# ### Create Derivative Dataframes

# #### 2018 Dataframe

# Get only 2018 data

df_2018 = df[df['sale_year'] == 2018]

# #### Agent Dataframe

# Calculate summary statistics

# Count
count_df_2018 = df_2018.groupby('agent_id')[["agent_id"]].count()
# Sum
sum_df_2018 = df_2018.groupby('agent_id')[['sale_amount', 'com_gross', 'com_agent_net','com_brokerage_net']].sum()
# Avg
mean_df_2018 = df_2018.groupby('agent_id')[['sale_amount', 'com_gross', 'com_agent_net','com_brokerage_net']].mean()
# Median
med_df_2018 = df_2018.groupby('agent_id')[['sale_amount', 'com_gross', 'com_agent_net','com_brokerage_net']].median()

# Rename columns for join

# +
count_df_2018 = count_df_2018.rename(columns={'agent_id': 'trans_count'})

sum_df_2018 = sum_df_2018.rename(columns={'sale_amount': 'sum_sales',
                                          'com_gross': 'sum_com_gross',
                                          'com_agent_net': 'sum_agent_com',
                                          'com_brokerage_net': 'sum_brokerage_com'})

mean_df_2018 = mean_df_2018.rename(columns={'sale_amount': 'avg_sales',
                                            'com_gross': 'avg_com_gross',
                                            'com_agent_net': 'avg_agent_com',
                                            'com_brokerage_net': 'avg_brokerage_com'})

med_df_2018 = med_df_2018.rename(columns={'sale_amount': 'med_sales',
                                          'com_gross': 'med_com_gross',
                                          'com_agent_net': 'med_agent_com',
                                          'com_brokerage_net': 'med_brokerage_com'})
# -

# Join into one dataframe

agent_df = count_df_2018.join(sum_df_2018).join(mean_df_2018).join(med_df_2018)

# Create a column for the quintiles the agents fall into

agent_df['quintile'] = pd.qcut(agent_df.sum_sales, 5, labels=["Bottom", "Below Average", "Average", "Above Average", "Top"])
agent_df['quintile_range'] = pd.qcut(agent_df.sum_sales, 5)

# Create a binary variable for top 20 percent

agent_df['top_quintile'] = (agent_df.quintile == 'Top').astype(int)

# Are any agent IDs connected to multiple accounts or brokerages?

print((df.groupby('agent_id').account_id.nunique() != 1).sum() > 0)
print((df.groupby('agent_id').brokerage_id.nunique() != 1).sum() > 0)

# Get the account and brokerage IDs for each agent. We use the mode because each agent ID is mapped to only one account_id and one brokerage_id, so the mode will suffice.

agent_df = agent_df.join(df_2018.groupby('agent_id')['account_id','brokerage_id'].agg(pd.Series.mode))

# How many transactions does each agent have under the two different commission plans?

com_plan_pivot = (df_2018.groupby(['agent_id', 'com_plan'])['com_plan']
                         .agg(pd.Series.value_counts)
                         .rename("count")
                         .reset_index()
                         .pivot(index="agent_id", columns="com_plan", values="count")
                         .fillna(0)
                         .astype(int))
com_plan_pivot.columns=["accum_strategy", "flat_strategy"]

agent_df = agent_df.join(com_plan_pivot)

# How many transactions does each agent have under the listing or selling side?

trans_side_pivot = (df_2018.groupby(['agent_id', 'trans_side'])['trans_side']
                           .agg(pd.Series.value_counts)
                           .rename("count")
                           .reset_index()
                           .pivot(index="agent_id", columns="trans_side", values="count")
                           .fillna(0)
                           .astype(int))
trans_side_pivot.columns=["selling_side", "buying_side"]

agent_df = agent_df.join(trans_side_pivot)

# ### Data Sanity/Validation Checks

df.isnull().sum()

# We could not find good ZIPs for these. We can leave them as NaN.

# Does agent_df contain only 2018 data?
# It must; otherwise, our models are learning from data used to derive our target variable

df[(df.sale_year == 2018)].sale_amount.sum() == agent_df.sum_sales.sum()

# Are there any transactions that have multiple agents on them??

(df.trans_id.value_counts() > 1).sum()

# Yes, there are quite a few.

# ## Exploration

# ### Summary Stats

# What are the dimensions?

df.shape

agent_df.shape

# How many transactions?

len(df_2018)

# ### By Agent

# Distribution of properties sold per agent

plt.figure(figsize=(12, 6))
sns.distplot(agent_df.trans_count, kde=False, rug=True, axlabel="Number of Transactions", color="green")
plt.ylabel("Number of Agents")
plt.title("Number of Transactions per Agent")
plt.show()

# Distribution of total sales by agent

plt.figure(figsize=(12, 6))
ax = sns.distplot(agent_df.sum_sales, kde=False, rug=True, axlabel="Total Sales", color="green")
plt.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel("Number of Agents")
plt.title("Total Sales per Agent")
plt.show()

# #### By Quintile

# What is the median number of transactions by quintile?

median_by_quintile = agent_df.groupby('quintile')['trans_count'].median()
plt.figure(figsize=(10, 6))
sns.barplot(x=median_by_quintile.index, y=median_by_quintile, color='green')
plt.xlabel("Rank")
plt.ylabel("Number of Transactions")
plt.title("Median Number of Transactions per Agent by Quintile")
plt.show()

# **Number of transactions is predictive of high performers**

# What is the median total sales by quintile?

median_by_quintile = agent_df.groupby('quintile')['sum_sales'].median()
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=median_by_quintile.index, y=median_by_quintile, color='green')
plt.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Rank")
plt.ylabel("Total Sales")
plt.title("Median Total Sales per Agent by Quintile")
plt.show()

# **Total sales is indictative of high performers**

# What is the minimum of total sales to be in each percentile?

min_by_quintile = agent_df.groupby('quintile')['sum_sales'].min()
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=min_by_quintile.index, y=min_by_quintile, color='green')
plt.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Rank")
plt.ylabel("Total Sales")
plt.title("Minimum Total Sales per Agent by Quintile")
plt.show()

# **An agent needs more than \$8mil at least to be in the top 20.**

# How many top-20s do we have?

agent_df.groupby('top_quintile')[['top_quintile']].count()

# What side are the agents on?

plt.figure(figsize=(10, 6))
num_agents_per_side = df_2018.groupby('trans_side')['agent_id'].count()
ax = sns.barplot(x=num_agents_per_side.index, y=num_agents_per_side, color='green')
plt.xlabel("Transaction Side")
plt.ylabel("Number of Agents")
plt.title("Number of Agents By Transaction Side")
plt.show()

# ### Sale Price/Commission

# What does the distribution of sale amount look like?

plt.figure(figsize=(12, 6))
ax = sns.distplot(df_2018.sale_amount, bins=100, kde=False, rug=True, axlabel='Sale Amount', color='green')
plt.ticklabel_format(style='plain', axis='x')
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('Number of Transactions')
plt.title('Distribution of Sale Amount')
plt.show()

# what is the average sale price by commission plan?

plt.figure(figsize=(10, 6))
num_agents_per_com_strategy = df_2018.groupby('com_plan')['sale_amount'].mean()
ax = sns.barplot(x=num_agents_per_com_strategy.index, y=num_agents_per_com_strategy, color='green')
plt.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Commission Strategy")
plt.ylabel("Price")
plt.title("Average Sale Price by Commission Strategy")
plt.show()

# How many agents are using the commission plans?

df_2018.groupby('com_plan')['agent_id'].count()

# **There's so few flat rate that it's pointless to make a feature out of this.**

# How do total sales look by month?

plt.figure(figsize=(15,10))
df.groupby(['sale_month', 'sale_year'])['sale_amount'].sum().plot.bar()
plt.show()

# We only have some of June 2019's data, and we excluded January 2018's data for incompleteness.

# Which side of the transaction makes the most money here?

# +
avg_sale_by_side = df.groupby('trans_side')['sale_amount'].mean()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=avg_sale_by_side.index, y=avg_sale_by_side)
plt.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Transaction Side")
plt.ylabel("Sale Amount")
plt.title("Mean Sale Price by Transaction Side")
plt.show()
# -

# What is the average commission based on the commission plan?

# +
avg_com_by_plan = df.groupby('com_plan')['com_gross'].mean()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=avg_com_by_plan.index, y=avg_com_by_plan)
plt.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Commission Plan")
plt.ylabel("Commission Amount")
plt.title("Mean Commission Amount by Commission Plan")
plt.show()
# -

# What commission plan takes more of the sale amount?

df.groupby('com_plan')['com_gross'].sum() / df.groupby('com_plan')['sale_amount'].sum()

# Accumulation strategy

# ### Feature Engineering

# #### Commission Strategy
# Find the most common commission strategy for each agent

for row in agent_df.iterrows():
    i = row[0]
    ser = row[1]
    D = dict(accum_strategy=ser.accum_strategy, flat_strategy=ser.flat_strategy)
    com_plan_sorted = sorted(D.items(), key=operator.itemgetter(1), reverse=True)
    agent_df.loc[i, 'com_plan_mode'] = com_plan_sorted[0][0]

agent_df.com_plan_mode.value_counts()

# Almost every agent at this brokerage is on the accumulation strategy so this distinction is not helpful at all.

# #### Add number of ZIP codes each agent sells in

agent_df['number_of_zips'] = df_2018.groupby('agent_id')['zip'].nunique()
agent_df.number_of_zips.fillna(1, inplace=True)

# #### Year over year difference in total sales
#
# What is the median percentage difference in total sales per agent from Feb-May 2018 to Feb-May 2019?
# This number will be used to adjust the $5 million threshold by 2019's price gains.

df_feb_may_2018 = df[(df.sale_date >= "2018-02-01") & (df.sale_date < "2018-06-01")][["agent_id", "sale_amount"]]
df_feb_may_2019 = df[(df.sale_date >= "2019-02-01") & (df.sale_date < "2019-06-01")][["agent_id", "sale_amount"]]

df_year_diff = pd.merge(df_feb_may_2018, df_feb_may_2019, how="inner", on="agent_id", suffixes=("_2018", "_2019"))
df_year_diff = df_year_diff.groupby("agent_id")[["sale_amount_2018", "sale_amount_2019"]].sum()
df_year_diff = df_year_diff.assign(year_diff=(df_year_diff.sale_amount_2019 - df_year_diff.sale_amount_2018) / df_year_diff.sale_amount_2018)
df_year_diff.year_diff.median()

# #### Calculate High Performance Threshold for Feb - May 2019
#
# What percent of 2018's total sales are Q1 and Q2 (excluding January because of its bad data)

total_sales_feb_may_2018 = df[(df['sale_date'] > '2018-01-31') & (df['sale_date'] < '2018-06-01')]['sale_amount'].sum()
total_sales_2018 = df[(df['sale_date'] > '2018-01-31') & (df['sale_date'] < '2019-01-01')]['sale_amount'].sum()

high_perf_thresh = (total_sales_feb_may_2018 / total_sales_2018) * 5_000_000 * 1.14

# Calculate the total sales for each agent for Feb. through May 2019 (excluding January)

agent_df['sales_sum_2019'] = df[(df['sale_date'] > '2019-01-31') & (df['sale_date'] < '2019-06-01')].groupby('agent_id')['sale_amount'].sum()

# Setting y to be the predicted

agent_df['y'] = (agent_df['sales_sum_2019'] > high_perf_thresh)
agent_df.dropna(subset=['sales_sum_2019'], inplace=True)

# #### Unique ZIP Codes per Agent

agent_df['number_of_zips'] = df.groupby('agent_id')['zip'].nunique()
agent_df.number_of_zips.fillna(1, inplace=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sum_sales', y='number_of_zips', hue='y', data=agent_df, alpha=0.5)
plt.xlabel("Sales Volume")
plt.ylabel("Number of Unique ZIP Codes")
plt.title("Agents' Number of Unique ZIP Codes Sold In According to Sales Volume")
plt.show()

# #### Agent cut v. Brokerage cut
# More experienced/successful agents tend to take a larger percentage of the commission

agent_df = agent_df.assign(agent_com_share=agent_df.sum_agent_com / (agent_df.sum_agent_com + agent_df.sum_brokerage_com))

# Average agent share of commission per quintile

# +
com_share_by_quintile = agent_df.groupby("quintile")["agent_com_share"].mean()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=com_share_by_quintile.index, y=com_share_by_quintile)
plt.xlabel("Quintile")
plt.ylabel("Commission Share")
plt.title("Agents' Mean Share of Commissions by Sales Volume Quintile")
plt.show()
# -

plt.figure(figsize=(10, 6))
sns.scatterplot(x=agent_df.sum_sales, y=agent_df.agent_com_share, hue=agent_df.y, alpha=0.5)
plt.xlabel('Sales Volume')
plt.ylabel('Commission Share')
plt.title("Agents' Share of Commissions by Sales Volume")
plt.show()

# Some agents' share is above 1.0 because the brokerage's commission was negative. This means the brokerage likely got its proceeds from the sale somewhere else.

# #### Number of Top ZIP Codes in Agent's Market

# How many zip codes have they sold in with a median home price in the top 20 percent of zip codes?

med_sale_by_zip = (df_2018.groupby("zip")['sale_amount']
                          .agg(['median', 'count']))
top_zips = med_sale_by_zip[med_sale_by_zip['median'] > med_sale_by_zip['median'].quantile(0.8)].index.values
df_2018 = df_2018.assign(top_zip=df_2018.zip.apply(lambda z: z in top_zips))
num_top_zip_by_agent = df_2018.groupby("agent_id")[['top_zip']].sum()
agent_df = pd.merge(agent_df, num_top_zip_by_agent, how='inner', on='agent_id')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sum_sales', y='top_zip', hue='y', data=agent_df, alpha=0.5)
plt.xlabel('Sales Volume')
plt.ylabel('Number of ZIP Codes')
plt.title("Agents' Number of High-priced ZIP Codes according to Sales Volume")
plt.show()

# **This is not a good feature. There are too many high performers that do not have many sales in the top ZIP codes in terms of sale price. The high-priced ZIPs are probably concentrated in specific cities and are not capturing agents' who sell elsewhere.**

# ### Proceeds

# Add column containing commission rate for each transaction

df['com_rate'] = df.com_gross / df.sale_amount

# What is the average commission rate for each commission plan?

df.groupby('com_plan')['com_rate'].mean()*100

# A higher share of the price for sales completed under a flat rate strategy go to agents and the brokerage.

# ### Statistical Tests

t_stat, p_val = stats.ttest_ind(agent_df[agent_df.y].trans_count, agent_df[~agent_df.y].trans_count)
print(f"T-stat: {t_stat}\np-val: {p_val}")

# We would reject the null hypothesis that there is not a significant difference in number of transactions completed between high performers and non-high performers.

t_stat, p_val = stats.ttest_ind(agent_df[agent_df.y].number_of_zips, agent_df[~agent_df.y].number_of_zips)
print(f"T-stat: {t_stat}\np-val: {p_val}")

# We would reject the null hypothesis that there is not a significant difference in number of unique ZIP codes in market between high performers and non-high performers.

# ## Modeling

agent_df.groupby('y')['y'].count() / len(agent_df)

# The classes are sufficiently balanced.

# ### Encoding Columns

for col in ('quintile',):
    le = LabelEncoder().fit(agent_df[col])
    label_encoders[col] = le
    agent_df[col] = le.transform(agent_df[col])

# ### Feature Selection

features = ['trans_count', 'sum_sales', 'number_of_zips']

# ### Train Test Split

# +
X = agent_df[features]
y = agent_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=agent_df.y, test_size = .2, random_state = 47)


# -

# ### Scale

# +
# Commented out to get a better visualization of the decision tree and for outputting model for web app

# scaler = RobustScaler().fit(X_train[['trans_count']])
# X_train[['trans_count']] = scaler.transform(X_train[['trans_count']])
# X_test[['trans_count']] = scaler.transform(X_test[['trans_count']])

# +
# scaler = RobustScaler().fit(X_train[['sum_sales']])
# X_train[['sum_sales']] = scaler.transform(X_train[['sum_sales']])
# X_test[['sum_sales']] = scaler.transform(X_test[['sum_sales']])

# +
# scaler = RobustScaler().fit(X_train[['number_of_zips']])
# X_train[['number_of_zips']] = scaler.transform(X_train[['number_of_zips']])
# X_test[['number_of_zips']] = scaler.transform(X_test[['number_of_zips']])
# -

# ### Cross Validation

# #### Logistic Regression

# No hyperparameter tuning

# +
logit = LogisticRegression(random_state=123)

print('Accuracy:', cross_val_score(logit, X_train, y_train, cv=3).mean())
print('Recall:', cross_val_score(logit, X_train, y_train, cv=3, scoring='recall').mean())
print('Precision:', cross_val_score(logit, X_train, y_train, cv=3, scoring='precision').mean())
print('F1:', cross_val_score(logit, X_train, y_train, cv=3, scoring='f1').mean())
# -

# #### Decision Tree

# No hyperparameter tuning

# +
tree = DecisionTreeClassifier(random_state=123)

print('Accuracy:', cross_val_score(tree, X_train, y_train, cv=3).mean())
print('Recall:', cross_val_score(tree, X_train, y_train, cv=3, scoring='recall').mean())
print('Precision:', cross_val_score(tree, X_train, y_train, cv=3, scoring='precision').mean())
print('F1:', cross_val_score(tree, X_train, y_train, cv=3, scoring='f1').mean())
# -

# Hyperparameter tuning

# +
tree = DecisionTreeClassifier(random_state=123, max_depth=3, class_weight='balanced')

print('Accuracy:', cross_val_score(tree, X_train, y_train, cv=3).mean())
print('Recall:', cross_val_score(tree, X_train, y_train, cv=3, scoring='recall').mean())
print('Precision:', cross_val_score(tree, X_train, y_train, cv=3, scoring='precision').mean())
print('F1:', cross_val_score(tree, X_train, y_train, cv=3, scoring='f1').mean())
# -

# Fit to training set

# +
tree.fit(X_train, y_train)
y_pred = tree.predict(X_train)
y_pred_proba = tree.predict_proba(X_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
 .format(tree.score(X_train, y_train)))
print('---')
print(classification_report(y_train, y_pred))
print('---')
# -

# What do the false negatives look like?

ser_pred = pd.Series(y_pred, index=y_train.index, name="pred")
tmp = pd.concat([y_train.rename("train"), ser_pred], axis=1)
bad_recall = tmp[(tmp.train) & (~tmp.pred)]
X_train.loc[bad_recall.index]

# #### Visualize Model

# +
import graphviz
from sklearn.tree import export_graphviz

class_names = ["False", "True"]

dot = export_graphviz(
    tree,
    out_file=None,
    feature_names=features,
    class_names=class_names, # target value names
    special_characters=True,
    filled=True,             # fill nodes w/ informative colors
    impurity=False,          # show impurity at each node
    leaves_parallel=True,    # all leaves at the bottom
    proportion=False,         # show percentages instead of numbers at each leaf
    rotate=False,             # top to bottom
    rounded=True,            # rounded boxes and sans-serif font
)

graph = graphviz.Source(dot, filename='decision_tree_viz', format='png')
# -

# #### Random Forest

# No hyperparameter tuning

# +
forest = RandomForestClassifier(random_state=123)

print('Accuracy:', cross_val_score(forest, X_train, y_train, cv=3).mean())
print('Recall:', cross_val_score(forest, X_train, y_train, cv=3, scoring='recall').mean())
print('Precision:', cross_val_score(forest, X_train, y_train, cv=3, scoring='precision').mean())
print('F1:', cross_val_score(forest, X_train, y_train, cv=3, scoring='f1').mean())
# -

# Hyperparameter tuning

# +
forest = RandomForestClassifier(random_state=123, n_estimators=100, max_depth=3, max_features=3, class_weight='balanced')

print('Accuracy:', cross_val_score(forest, X_train, y_train, cv=3).mean())
print('Recall:', cross_val_score(forest, X_train, y_train, cv=3, scoring='recall').mean())
print('Precision:', cross_val_score(forest, X_train, y_train, cv=3, scoring='precision').mean())
print('F1:', cross_val_score(forest, X_train, y_train, cv=3, scoring='f1').mean())
# -

# #### KNN

# +
knn = KNeighborsClassifier()

print('Accuracy:', cross_val_score(knn, X_train, y_train, cv=3).mean())
print('Recall:', cross_val_score(knn, X_train, y_train, cv=3, scoring='recall').mean())
print('Precision:', cross_val_score(knn, X_train, y_train, cv=3, scoring='precision').mean())
print('F1:', cross_val_score(knn, X_train, y_train, cv=3, scoring='f1').mean())
# -
# ### Grid Search

# The code for each learning method is commented out because of it is CPU-intensive. Accordingly, it will not run by default. Uncomment to run.

metrics = ['accuracy', 'precision', 'recall']

# #### Logistic Regression

# +
# hyperparameters = {
#     'class_weight': ['balanced', None],
#     'random_state': [47],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# }

# grid = GridSearchCV(LogisticRegression(), param_grid=hyperparameters, cv=3, scoring=metrics, refit=False, n_jobs=5)
# grid.fit(X_train, y_train)

# results = grid.cv_results_
# accuracies = results['mean_test_accuracy']
# precisions = results['mean_test_precision']
# recalls = results['mean_test_recall']
# params = results['params']

# for acc, prec, rec, par in zip(accuracies, precisions, recalls, params):
#     par['accuracy'] = acc
#     par['precision'] = prec
#     par['recall'] = rec

# pd.DataFrame(params).sort_values(by=['recall', 'precision'], ascending=False).head()
# -

# #### Decision Tree

# +
# hyperparameters = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': list(range(1, 11)),
#     'max_features': list(range(1, 5)),
#     'min_samples_split': [2, 5, 10, 15, 20, 30],
#     'min_samples_leaf': [1, 2, 3, 5, 10, 15, 20],
#     'random_state': [47],
#     'class_weight': [None, 'balanced']
# }

# grid = GridSearchCV(DecisionTreeClassifier(), param_grid=hyperparameters, cv=3, scoring=metrics, refit=False, n_jobs=5)
# grid.fit(X_train, y_train)

# results = grid.cv_results_
# accuracies = results['mean_test_accuracy']
# precisions = results['mean_test_precision']
# recalls = results['mean_test_recall']
# params = results['params']

# for acc, prec, rec, par in zip(accuracies, precisions, recalls, params):
#     par['accuracy'] = acc
#     par['precision'] = prec
#     par['recall'] = rec

# pd.DataFrame(params).sort_values(by=['recall', 'precision'], ascending=False).head()
# -

# #### Random Forest

# +
# hyperparameters = {
#     'nestimators': [10, 100]
#     'criterion': ['gini', 'entropy'],
#     'max_depth': list(range(1, 11)),
#     'max_features': list(range(1, 5)),
#     'min_samples_split': [2, 5, 10, 15, 20, 30],
#     'min_samples_leaf': [1, 2, 3, 5, 10, 15, 20],
#     'random_state': [47],
#     'class_weight': [None, 'balanced'],
# }



# grid = GridSearchCV(RandomForestClassifier(), param_grid=hyperparameters, cv=3, scoring=metrics, refit=False, n_jobs=5)
# grid.fit(X_train, y_train)

# results = grid.cv_results_
# accuracies = results['mean_test_accuracy']
# precisions = results['mean_test_precision']
# recalls = results['mean_test_recall']
# params = results['params']

# for acc, prec, rec, par in zip(accuracies, precisions, recalls, params):
#     par['accuracy'] = acc
#     par['precision'] = prec
#     par['recall'] = rec

# pd.DataFrame(params).sort_values(by=['recall', 'precision'], ascending=False).head()
# -

# ### Final Model

# +
forest = RandomForestClassifier(max_depth=2,
                                criterion='entropy',
                                max_features=3, 
                                random_state=123)

forest.fit(X_train, y_train)

print(X_train.columns)
print(forest.feature_importances_)

y_pred = forest.predict(X_train)
y_pred_test = forest.predict(X_test)

# + {"endofcell": "--"}
# Performance Metrics

print('Train')

print('---'*20)

print('Accuracy of random forest classifier on training set: {:.2f}'
     .format(forest.score(X_train, y_train)))

print('---'*20)

print(confusion_matrix(y_train, y_pred))

print('---'*20)

print(classification_report(y_train, y_pred))

print('---'*20)

# -

# ### Modeling Out of Sample Data

# # +
print('Test')

print('---'*20)

print('Accuracy of random forest classifier on testing set: {:.2f}'
     .format(forest.score(X_test, y_test)))

print('---'*20)

print(confusion_matrix(y_test, y_pred_test))

print('---'*20)

print(classification_report(y_test, y_pred_test))

print('---'*20)
# -
# --

# #### Pickle Model for Web App

with open('rf_model_agents.obj', 'wb') as fp:
    pickle.dump(forest, fp)

# #### Test Pickled Model

# +
with open('rf_model_agents.obj', 'rb') as fp:
    rf_model_agents = pickle.load(fp)

# # +
agent_cols = features
agent_info = [10, 10_000_000, 15]

agent_dict = dict(zip(agent_cols, agent_info))
agent_info = pd.DataFrame(agent_dict, index=[1])

output = rf_model_agents.predict(agent_info)
if output[0]:
    prediction = 'Agent is trending to be a $5mil+ agent.'
else:
    prediction = 'Agent is not trending to be a $5mil+ agent.'
print(prediction)
