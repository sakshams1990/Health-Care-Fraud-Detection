import pandas as pd
import numpy as np
from warnings import filterwarnings

df = pd.read_csv("../Fraud Detection/Dataset/Final/fraud_model.csv")

df_num = df[['Claim Amount', 'IP Sum Assured', 'IP Annual Premium',
             'OP Sum Assured', 'OP Annual Premium', 'Age',
             'Claim Duration', 'Potential Fraud']]

# df_fraud = df_num[df_num['Potential Fraud'] == 1]

# df_non_fraud = df_num[df_num['Potential Fraud'] == 0]

df['op_annual_premium_range'] = pd.cut(np.array(df_num['OP Annual Premium']), bins=[*range(0, 18000, 2000)],
                                       labels=[*range(1, 9, 1)])

# count = (df_non_fraud['age_range'].value_counts()/(df_num[df_num['Potential Fraud'] == 0]))*100
# print(count)

val_cnt = (df['op_annual_premium_range'].groupby(df['Potential Fraud'] == 1).value_counts() / (
    df.groupby(['op_annual_premium_range'])['Potential Fraud'].count())) * 100

print(df['op_annual_premium_range'])
