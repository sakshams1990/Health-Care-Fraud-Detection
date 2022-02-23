import json
import pickle
from configparser import ConfigParser, ExtendedInterpolation

import pandas as pd


# Reading all the objects under the section DATA MODEL
def get_dataset_filepath_from_config():
    config = ConfigParser()
    config._interpolation = ExtendedInterpolation()
    config.read('Config.ini')
    pred_dataset_value_dict = dict(config.items('DATASET'))
    return pred_dataset_value_dict


# Reading all the objects from the dictionary obtained from get dataset details from config
def read_dataset():
    read_datasets = get_dataset_filepath_from_config()
    # Reading train data sets
    ben = pd.read_csv(read_datasets['train_ben'])
    inpatient = pd.read_csv(read_datasets['train_in'])
    outpatient = pd.read_csv(read_datasets['train_out'])
    fraud = pd.read_csv(read_datasets['train_fraud'])
    return ben, inpatient, outpatient, fraud


# Merging all the dataset to create one dataframe
def merging_dataset():
    train_ben, train_in, train_out, train_fraud = read_dataset()

    # Creating a new column called Admission Type. Admission Type is 1 for Inpatient and 0 for Outpatient
    train_in['Admission Type'] = 1
    train_out['Admission Type'] = 0

    # Merging Inpatient and Outpatient columns based on common columns without dropping any rows.
    train_merge = pd.merge(train_in, train_out, left_on=[x for x in train_out.columns if x in train_in.columns],
                           right_on=[x for x in train_out.columns if x in train_in], how='outer')

    # Merging the in-out patient data with benefit dataset
    train_merge_ben = pd.merge(train_merge, train_ben, left_on='BeneID', right_on='BeneID', how='inner')

    # Merging fraud detail dataframe to train_all and test_all dataset
    train_all = pd.merge(train_merge_ben, train_fraud, on='Provider')

    data_types = train_all.dtypes.astype(str).to_dict()
    with open('../Fraud Detection/Dataset/Merged Raw/data_type_model.pkl', 'wb') as f:
        pickle.dump(data_types, f)
    train_all.to_csv('../Fraud Detection/Dataset/Merged Raw/raw_df.csv', index=False)
    return train_all


# Performing feature engineering
def feature_engineering():
    fraud_df = merging_dataset()
    # For Chronic diseases, replacing 2 with 0. 0 indicates No and 1 indicates as Yes
    fraud_df = fraud_df.replace(
        {'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
         'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2,
         'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2,
         'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2}, 0)
    # For Renal disease , replacing Y with 1
    fraud_df = fraud_df.replace({'RenalDiseaseIndicator': 'Y'}, 1)
    fraud_df['State'] = fraud_df['State'].astype('str')
    fraud_df['State'].replace({'39': 'Alabama', '52': 'Alaska', '24': 'Arizona', '23': 'Arkansas', '45': 'California',
                               '15': 'Colorado', '44': 'Connecticut', '41': 'Delaware', '1': 'District of Columbia',
                               '14': 'Florida', '7': 'Georgia', '13': 'Hawaii', '34': 'Idaho', '31': 'Illinois',
                               '5': 'Indiana', '49': 'Iowa', '46': 'Kansas', '6': 'Kentucky', '38': 'Louisiana',
                               '10': 'Maine', '26': 'Maryland', '3': 'Massachusetts', '20': 'Michigan',
                               '33': 'Minnesota', '18': 'Mississippi', '27': 'Missouri', '51': 'Montana',
                               '29': 'Nebraska', '42': 'Nevada', '36': 'New Hampshire', '11': 'New Jersey',
                               '12': 'New Mexico', '30': 'New York', '19': 'North Carolina', '4': 'North Dakota',
                               '25': 'Ohio', '17': 'Oklahoma', '16': 'Oregon', '32': 'Pennsylvania',
                               '2': 'Rhode Island', '21': 'South Carolina', '50': 'South Dakota', '22': 'Tennessee',
                               '54': 'Texas', '28': 'Utah', '43': 'Vermont', '35': 'Virginia', '37': 'Washington',
                               '47': 'West Virginia', '8': 'Wisconsin', '9': 'Wyoming', '53': 'Guam'}, inplace=True)
    # Dropping the Claim Procedure Codes and Claim Diagnostic Codes features
    fraud_df = fraud_df.drop(['ClmDiagnosisCode_1',
                              'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                              'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
                              'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
                              'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
                              'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6'], axis=1)

    # Creating a list of columns whose datatype need to be changed to Datetime
    cols_date = ['DOB', 'DOD', 'ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt']
    # Changing the datatype of columns to datetime and categorical for train and test
    fraud_df[cols_date] = fraud_df[cols_date].apply(pd.to_datetime, format='%Y-%m-%d')

    # Calculating Age from DOD and DOB and converting it into number of years and rounding it to have 0 decimal points
    fraud_df['Age'] = round((fraud_df['DOD'] - fraud_df['DOB']).dt.days / 365)
    # Last DOD is of 1st Dec 2009, hence the dataset if of 2009 and we can calculate age of those who are still alive
    fraud_df['Age'].fillna(round((pd.to_datetime('2009-12-31', format='%Y-%m-%d') - fraud_df['DOB']).dt.days / 365),
                           inplace=True)

    # Calculating Claim Duration from ClaimStartDt and ClaimEndDt and converting it into number of days
    # With the claims having start date and end date as same day, showing it as 1
    fraud_df['Claim Duration'] = (fraud_df['ClaimEndDt'] - fraud_df['ClaimStartDt']).dt.days + 1

    # Dropping the date columns and Diagnosis Group Codes from train and test
    fraud_df = fraud_df.drop(['DOB', 'DOD', 'ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt',
                              'DiagnosisGroupCode'], axis=1)

    # Creating a list of columns whose datatype need to be changed to Categorical
    cols_float = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
                  'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt']
    fraud_df[cols_float] = fraud_df[cols_float].astype('float')

    # Creating a list of columns whose datatype needs to be changed to Integer
    cols_int = 'Age'
    fraud_df[cols_int] = fraud_df[cols_int].astype('int64')

    # Filling OTHERS for missing values in categorical variables & median in numerical variable
    fraud_df['OperatingPhysician'] = fraud_df['OperatingPhysician'].fillna('OTHERS')
    fraud_df['AttendingPhysician'] = fraud_df['AttendingPhysician'].fillna('OTHERS')
    fraud_df['DeductibleAmtPaid'].fillna(fraud_df['DeductibleAmtPaid'].median(), inplace=True)

    # Replacing Yes with 1 and No with 0 for Potential Fraud column
    pot_fraud = {'Yes': 1, 'No': 0}
    fraud_df['PotentialFraud'] = [pot_fraud[item] for item in fraud_df['PotentialFraud']]

    fraud_df = fraud_df.drop(['ClmAdmitDiagnosisCode', 'Race', 'Gender', 'BeneID', 'County', 'ClaimID',
                              'OtherPhysician', 'ClmAdmitDiagnosisCode', 'NoOfMonths_PartACov',
                              'NoOfMonths_PartBCov', 'DeductibleAmtPaid'], axis=1)

    data_types = fraud_df.dtypes.astype(str).to_dict()
    with open('../Fraud Detection/Dataset/DataViz/data_type_model.pkl', 'wb') as f:
        pickle.dump(data_types, f)
    fraud_df.to_csv('../Fraud Detection/Dataset/DataViz/fraud_viz_df.csv', index=False)

    # Fraud Potential for all the categorical variables
    fraud_pot_cat = ['Provider', 'AttendingPhysician', 'OperatingPhysician', 'State', 'ChronicCond_Alzheimer',
                     'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                     'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes',
                     'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                     'ChronicCond_stroke', 'Admission Type', 'RenalDiseaseIndicator']

    for col in fraud_pot_cat:
        count = (fraud_df[col].groupby(fraud_df['PotentialFraud']).value_counts() /
                 (fraud_df.groupby([col])['PotentialFraud'].count())) * 100
        count_dict = dict(count[1])
        with open(f"../Fraud Detection/Fraud Potential/{col}.json", "w") as fraud_count:
            json.dump(count_dict, fraud_count, indent=4)

    # High Cardinality feature creation
    card_cols = ['Health Care Provider', 'Attending Physician', 'Operating Physician', 'State']
    for col in card_cols:
        fe = fraud_df.groupby(col).size()
        fraud_df.loc[:, col + '_freqencode'] = fraud_df[col].map(fe)

    # Creating Pickle files for Provider, Attending Physician, Operating Physician and State
    provider_codes = dict(fraud_df['Health Care Provider'].value_counts())
    attendphy_codes = dict(fraud_df['Attending Physician'].value_counts())
    operatephy_codes = dict(fraud_df['Operating Physician'].value_counts())
    state_codes = dict(fraud_df['State'].value_counts())

    # Dropping columns that would not be required for model building
    fraud_df = fraud_df.drop(['Provider', 'Attending Physician', 'Operating Physician', 'State'], axis=1)

    # Creating pickle files for provider, attending physician, operating physicians and state
    with open("../Fraud Detection/Trained Model/Resources/provider_codes.pkl", "wb") as provider:
        pickle.dump(provider_codes, provider)
    with open("../Fraud Detection/Trained Model/Resources/attendphy_codes.pkl", "wb") as attend:
        pickle.dump(attendphy_codes, attend)
    with open("../Fraud Detection/Trained Model/Resources/operatephy_codes.pkl", "wb") as operate:
        pickle.dump(operatephy_codes, operate)
    with open("../Fraud Detection/Trained Model/Resources/state_codes.pkl", "wb") as state:
        pickle.dump(state_codes, state)

    # Renaming the column names as per UI
    fraud_df.rename(columns={'Health Care Provider_freqencode': 'Health Care Provider',
                             'Attending Physician_freqencode': 'Attending Physician',
                             'Operating Physician_freqencode': 'Operating Physician',
                             'State_freqencode': 'State',
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

    # Creating fraud column names dataframe to get
    fraud_column_names = fraud_df.copy()
    fraud_column_names = fraud_column_names.drop(['Potential Fraud'], axis=1)
    column_names = fraud_column_names.columns
    with open('../Fraud Detection/Trained Model/Resources/model_columns.pkl', 'wb') as cols:
        pickle.dump(column_names, cols)

    # Storing the features into new csv file
    data_types = fraud_df.dtypes.astype(str).to_dict()
    with open('../Fraud Detection/Dataset/Final/data_type_model.pkl', 'wb') as f:
        pickle.dump(data_types, f)
    final_df = fraud_df.to_csv('../Fraud Detection/Dataset/Final/fraud_model.csv', index=False)
    return final_df


# Calling feature engineering function to initiate the data cleaning and feature engineering process
if __name__ == '__main__':
    df = feature_engineering()
