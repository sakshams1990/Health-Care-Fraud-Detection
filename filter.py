import itertools
import re
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


# Read the dataframe
def read_dataset(path):
    data = pd.read_csv(path)
    return data


# Creating bins for numerical features in the dataframe
def create_numerical_bins(dataset):
    dataset["Age Labels"] = pd.cut(np.array(dataset['Age']), bins=[*range(0, 120, 10)])
    dataset["Claim Duration Labels"] = pd.cut(np.array(dataset['Claim Duration']),
                                              bins=[*range(0, 50, 5)])
    dataset["Claim Amount Labels"] = pd.cut(np.array(dataset['Claim Amount']), bins=np.arange(0.0, 165000.0, 15000.0))
    dataset["IP Sum Assured Labels"] = pd.cut(np.array(dataset['IP Sum Assured']),
                                              bins=np.arange(-20000.0, 220000.0, 20000.0))
    dataset["OP Sum Assured Labels"] = pd.cut(np.array(dataset['OP Sum Assured']),
                                              bins=np.arange(-20000.0, 160000.0, 20000.0))
    dataset["IP Annual Premium Labels"] = pd.cut(np.array(dataset['IP Annual Premium']),
                                                 bins=np.arange(-5000.0, 50000.0, 5000.0))
    dataset["OP Annual Premium Labels"] = pd.cut(np.array(dataset['OP Annual Premium']),
                                                 bins=np.arange(-2000.0, 18000.0, 2000.0))
    return dataset


def preprocess_dataframe(dataset):
    dataset.rename(columns={'Provider': 'Health Care Provider',
                            'AttendingPhysician': 'Attending Physician',
                            'OperatingPhysician': 'Operating Physician',
                            'State': 'State',
                            'ChronicCond_Alzheimer': 'Alzheimer', 'ChronicCond_Heartfailure': 'Heart Failure',
                            'ChronicCond_KidneyDisease': 'Kidney Disease', 'ChronicCond_Cancer': 'Cancer',
                            'ChronicCond_ObstrPulmonary': 'Obstruction Pulmonary',
                            'ChronicCond_Depression': 'Depression', 'ChronicCond_Diabetes': 'Diabetes',
                            'ChronicCond_IschemicHeart': 'Ischemic Heart', 'ChronicCond_Osteoporasis': 'Osteoporasis',
                            'ChronicCond_rheumatoidarthritis': 'Rheumatoid Arthritis', 'ChronicCond_stroke': 'Stroke',
                            'InscClaimAmtReimbursed': 'Claim Amount', 'IPAnnualReimbursementAmt': 'IP Sum Assured',
                            'IPAnnualDeductibleAmt': 'IP Annual Premium', 'OPAnnualReimbursementAmt': 'OP Sum Assured',
                            'OPAnnualDeductibleAmt': 'OP Annual Premium', 'PotentialFraud': 'Potential Fraud'},
                   inplace=True)

    cols = ['RenalDiseaseIndicator', 'Alzheimer', 'Heart Failure', 'Kidney Disease',
            'Cancer', 'Obstruction Pulmonary', 'Depression', 'Diabetes', 'Ischemic Heart', 'Osteoporasis',
            'Rheumatoid Arthritis', 'Stroke']

    for col in cols:
        dataset[col] = dataset[col].map({0: 'False', 1: 'True'})

    dataset['Admission Type'] = dataset['Admission Type'].map({0: 'O', 1: 'I'})
    return dataset


def drop_rename_cols(data):
    data = data.drop(['Claim Amount', 'IP Sum Assured', 'IP Annual Premium', 'OP Sum Assured', 'OP Annual Premium',
                      'Age', 'Claim Duration'], axis=1)

    data.rename(columns={'Claim Amount Labels': 'Claim Amount', 'IP Sum Assured Labels': 'IP Sum Assured',
                         'IP Annual Premium Labels': 'IP Annual Premium', 'OP Sum Assured Labels': 'OP Sum Assured',
                         'OP Annual Premium Labels': 'OP Annual Premium', 'Age Labels': 'Age',
                         'Claim Duration Labels': 'Claim Duration'}, inplace=True)

    for col in data.columns:
        data[col] = data[col].astype(str)

    return data


# Preprocess Causality items
def preprocess_causality_dic(dic):
    num_dict = {'Age': 10, 'Claim Duration': 5, 'Claim Amount': 15000.0, 'IP Sum Assured': 20000.0,
                'OP Sum Assured': 20000.0, 'IP Annual Premium': 5000.0, 'OP Annual Premium': 2000.0}
    preprocess_dic = {}
    for key in dic.keys():
        value = [part.strip() for part in re.split('[\(\)]', key)]
        card_cols = ['Admission Type', 'Health Care Provider', 'State', 'Attending Physician', 'Operating Physician',
                     'RenalDiseaseIndicator', 'Alzheimer', 'Heart Failure', 'Kidney Disease', 'Cancer', 'Depression',
                     'Obstruction Pulmonary', 'Depression', 'Diabetes', 'Ischemic Heart', 'Osteoporasis',
                     'Rheumatoid Arthritis', 'Stroke']
        if value[0] in card_cols:
            preprocess_dic[value[0]] = value[1]

        else:
            range_min = int(float(value[1]) / num_dict[value[0]]) * num_dict[value[0]]
            range_max = range_min + num_dict[value[0]]
            bin_range = f"({range_min}, {range_max}]"
            preprocess_dic[value[0]] = bin_range

    return preprocess_dic


# Consider the first two returned valid items in the dictionary
def filter_dic_keys(dictionary):
    if len(dictionary) >= 2:
        final_dict = dict(itertools.islice(dictionary.items(), 2))
        return final_dict


# Filter dataframe using keys as column name and value as value in the column
def filter_dataframe(data, dic):
    if len(dic) >= 2:
        filter_df = data.loc[data[dic.keys()].isin(dic.values()).all(axis=1), :]
        return filter_df
    else:
        print('No item present in dictionary')


def generate_keys(dic):
    return " and ".join(f"{key} - {value}" for key, value in dic.items())


def calculate_fraud_perc(fraud_df):
    if len(fraud_df) > 0:
        return round(fraud_df['Potential Fraud'].value_counts()[1] / len(fraud_df) * 100, 2)
    else:
        return 0


def calculate_non_fraud_perc(non_fraud_df):
    if len(non_fraud_df) > 0:
        percent = 100 - round(non_fraud_df['Potential Fraud'].value_counts()[0] / len(non_fraud_df) * 100, 2)
        return percent
    else:
        return 0


def generate_text(keys, percentage):
    txt = "Claims with " + str(keys) + " are involved in " + str(percentage) + "% fraudulent activities"
    txt = txt.replace(']', ')')
    return txt


def final_response(causality_dict):
    df = read_dataset("./Dataset/DataViz/fraud_viz_df.csv")
    df = preprocess_dataframe(df)
    df = create_numerical_bins(df)
    df = drop_rename_cols(df)

    fraud_y = preprocess_causality_dic(causality_dict["fraud_reported_y"])
    fraud_n = preprocess_causality_dic(causality_dict["fraud_reported_n"])

    fraud = filter_dic_keys(fraud_y)
    non_fraud = filter_dic_keys(fraud_n)

    fraud_filter_df = filter_dataframe(df, fraud)
    non_fraud_filter_df = filter_dataframe(df, non_fraud)

    fraud_keys = generate_keys(fraud)
    non_fraud_keys = generate_keys(non_fraud)

    fraud_percent = calculate_fraud_perc(fraud_filter_df)
    non_fraud_percentage = calculate_non_fraud_perc(non_fraud_filter_df)

    fraud_text = generate_text(fraud_keys, fraud_percent)
    non_fraud_text = generate_text(non_fraud_keys, non_fraud_percentage)

    return fraud_text, non_fraud_text


if __name__ == '__main__':
    causality = {
        "fraud_reported_y": {
            "Admission Type (I)": -0.47313711047172546,
            "Claim Amount (9000.0)": -0.274419903755188,
            "Claim Duration (3)": -0.09963756799697876,
            "Ischemic Heart (False)": -0.04432913288474083
        },
        "fraud_reported_n": {
            "Health Care Provider (Saksham)": 0.8289695978164673,
            "Attending Physician (Saksham)": 0.12608131766319275,
            "State (Oklahoma)": 0.09439872205257416,
            "Operating Physician (Saksham)": 0.06448725610971451

        }
    }
    fraud_txt, non_fraud_txt = final_response(causality)
    print(fraud_txt)
    print(non_fraud_txt)
