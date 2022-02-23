import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import dataset
with open("../Fraud Detection/Dataset/DataViz/data_type_model.pkl", "rb") as dt:
    datatype = pickle.load(dt)
df_data_file = pd.read_csv('../Fraud Detection/Dataset/DataViz/fraud_viz_df.csv', dtype=datatype)
'''
fraud_pot_cat = ['Provider', 'AttendingPhysician', 'OperatingPhysician', 'State', 'ChronicCond_Alzheimer',
                 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes',
                 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                 'ChronicCond_stroke', 'Admission Type', 'RenalDiseaseIndicator']


for column in column1:
    sns.set_style('white', rc={'figure.figsize': (10, 8)})
    g = sns.catplot(x=column, hue='PotentialFraud', data=fraud,kind='count',
                        order=fraud[column].value_counts().index)
    g.fig.subplots_adjust(top=0.95)
    g.ax.set_title(column)
    plt.xlabel("Classes for {}".format(column))
    plt.ylabel("Number of Fraud/Non Fraud Cases ")
    admission_type = len(fraud[column])
    for p in g.ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / admission_type)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        value = int(p.get_height())
        g.ax.annotate(percentage,(x, y),ha='center')
    plt.savefig('Data Visualizations/Fraud Detection based on {}'.format(column))

for val in fraud_pot_cat:
    sns.set_style('white', rc={'figure.figsize': (10, 8)})
    g = sns.catplot(y=val, hue='PotentialFraud', data=df_data_file, kind='count',
                    order=df_data_file[val].value_counts()[:20].index)
    g.fig.subplots_adjust(top=0.95)
    g.ax.set_title(val)
    plt.xlabel("Classes for {}".format(val))
    plt.ylabel("Number of Fraud/Non Fraud Cases ")
    total = len(df_data_file[val])
    for p in g.ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width() / total)
        x = p.get_x() + p.get_width() + 0.04
        y = p.get_y() + p.get_height()
        g.ax.annotate(percentage, (x, y), ha='center')
    plt.savefig('../Fraud Detection/Data Visualization/Fraud Detection based on {}'.format(val))


cols_int = ['InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
            'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
            'Age', 'Claim Duration']

sns.histplot(data=df_data_file, x='InscClaimAmtReimbursed', hue='PotentialFraud', bins = 10)
plt.savefig(f"../Fraud Detection/Data Visualization/InscClaimAmtReimbursed.png")
'''

fig, ax1 = plt.subplots(figsize=(20, 10))
plt.xlabel('Potential Fraud', fontsize=18)
plt.ylabel('Count', fontsize=16)
graph = sns.countplot(ax=ax1, x='PotentialFraud', data=df_data_file)
graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,
        df_data_file['PotentialFraud'].value_counts()[i], ha="center")
    i += 1
plt.savefig(f"../Fraud Detection/Data Visualization/FraudCount.png")


