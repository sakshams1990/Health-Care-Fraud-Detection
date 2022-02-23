import itertools
import json
import operator as op
import os
import re

import Utils_Configurations
from Utils import sort_shap_value

from filter import final_response

FRAUD_EXPLANATION = "fraud_explanation"
NON_FRAUD_EXPLANATION = "non_fraud_explanation"


# Fetching number of causalities from config.ini
def get_default_values_from_configuration():
    config_file_path = os.path.join(os.getcwd(), "Config.ini")
    config = Utils_Configurations.Configuration(config_file_path)
    num_of_causalities = config.read_configuration_options("CAUSALITY", "number_of_causalities", "int")
    pred_thresh = config.read_configuration_options("CAUSALITY", "prediction_threshold", "float")
    num_of_decimals = config.read_configuration_options("CAUSALITY", "fraudScore_Number_of_Decimals", "int")
    fraud_file = config.read_configuration_options("FRAUD TEXT", "fraud")
    non_fraud_file = config.read_configuration_options("FRAUD TEXT", "non_fraud")
    flag = config.read_configuration_options("COMPLEX FILTER", "flag")
    return num_of_causalities, pred_thresh, num_of_decimals, fraud_file, non_fraud_file, flag


# Initializing labels and causalties category
labels = ["fraud_reported_n", "fraud_reported_y"]
number_of_causalities, prediction_threshold, fraudScore_Number_of_Decimals, \
    fraud, non_fraud, filter_flag = get_default_values_from_configuration()


# Normalizing positive values in the dictionary
def normalize_shap_pos_values(dic):
    return {k: v / max(dic.values()) * 100 for k, v in dic.items()}


# Normalizing negative values in the dictionary
def normalize_shap_neg_values(dic):
    return {k: v / min(dic.values()) * (-100) for k, v in dic.items()}


# Converting the shap dictionary into 2 separate dictionaries to contain positive and negative shap values
# Normalizing both the dictionaries and sorting the shap values in descending value
def get_casuality_data(shap_dic):
    keys = list(shap_dic.keys())
    if 'Admission Type (I)' in shap_dic.keys():
        for key in keys:
            if key.startswith('OP '):
                shap_dic.pop(key)
    else:
        for key in keys:
            if key.startswith('IP '):
                shap_dic.pop(key)

    causality = {}
    shap_dict_pos = {}  # Positive Shap Values
    shap_dict_neg = {}  # Negative Shap Values
    for (key, value) in shap_dic.items():  # For loop to save values in pos & neg dict
        if op.lt(value, 0):  # Using operator package for less than. Imported operator as op package.lt stands for
            # less than
            shap_dict_neg[key] = value
        else:
            shap_dict_pos[key] = value
    # shap_norm_pos = normalize_shap_pos_values(shap_dict_pos)  # Normalizing the values
    # shap_norm_neg = normalize_shap_neg_values(shap_dict_neg)  # Normalizing the values
    causality["fraud_reported_y"] = dict(itertools.islice(sort_shap_value(shap_dict_pos, True).items(),
                                                          number_of_causalities))
    causality["fraud_reported_n"] = dict(itertools.islice(sort_shap_value(shap_dict_neg, False).items(),
                                                          number_of_causalities))
    # Multiplying the causality values with -1
    for key in causality["fraud_reported_y"]:
        causality["fraud_reported_y"][key] = causality["fraud_reported_y"][key] * (-1)
    for key in causality["fraud_reported_n"]:
        causality["fraud_reported_n"][key] = causality["fraud_reported_n"][key] * (-1)
    return causality


def get_text_explanation(causality_dict):
    # Response of fraud_reported_n and fraud_reported_y goes into Fraud Explanation and Non Fraud Explanation
    fraud_response = {FRAUD_EXPLANATION: [], NON_FRAUD_EXPLANATION: []}
    num_dict = {'Age': 10, 'Claim Duration': 5, 'Claim Amount': 15000.0, 'IP Sum Assured': 20000.0,
                'OP Sum Assured': 20000.0, 'IP Annual Premium': 5000.0, 'OP Annual Premium': 2000.0}
    fraud_txt, non_fraud_txt = final_response(causality_dict)
    with open(fraud) as f:

        fraud_text = json.load(f)
    with open(non_fraud) as nf:
        non_fraud_text = json.load(nf)

    # Fetch all values for the keys present in the fraud file
    if filter_flag == 'Y':
        fraud_response[FRAUD_EXPLANATION].append(fraud_txt)
    for key in causality_dict["fraud_reported_y"].keys():
        if key in fraud_text:
            fraud_response[FRAUD_EXPLANATION].append(fraud_text[key])
        elif key.startswith('Health'):
            value = [part.strip() for part in re.split('[\(\)]', key)]
            fraud_response[FRAUD_EXPLANATION].append(fraud_text['Health Care Provider (Unknown)'])
            fraud_response[FRAUD_EXPLANATION] = list(map(lambda st: str.replace(st, "Unknown", value[1]),
                                                         fraud_response[FRAUD_EXPLANATION]))
        elif key.startswith('Attending'):
            value = [part.strip() for part in re.split('[\(\)]', key)]
            fraud_response[FRAUD_EXPLANATION].append(fraud_text['Attending Physician (Unknown)'])
            fraud_response[FRAUD_EXPLANATION] = list(map(lambda st: str.replace(st, "Unknown", value[1]),
                                                         fraud_response[FRAUD_EXPLANATION]))
        elif key.startswith('Operating'):
            value = [part.strip() for part in re.split('[\(\)]', key)]
            fraud_response[FRAUD_EXPLANATION].append(fraud_text['Operating Physician (Unknown)'])
            fraud_response[FRAUD_EXPLANATION] = list(map(lambda st: str.replace(st, "Unknown", value[1]),
                                                         fraud_response[FRAUD_EXPLANATION]))
        else:
            value = [part.strip() for part in re.split('[\(\)]', key)]
            range_min = int(float(value[1]) / num_dict[value[0]]) * num_dict[value[0]]
            range_max = range_min + num_dict[value[0]]
            json_key = f"{value[0]} ({range_min}-{range_max - 1})"
            if json_key in fraud_text.keys():
                fraud_response[FRAUD_EXPLANATION].append(fraud_text[json_key])

    if filter_flag == 'Y':
        fraud_response[NON_FRAUD_EXPLANATION].append(non_fraud_txt)
    for key in causality_dict["fraud_reported_n"].keys():
        if key in non_fraud_text:
            fraud_response[NON_FRAUD_EXPLANATION].append(non_fraud_text[key])
        elif key.startswith('Health'):
            value = [part.strip() for part in re.split('[\(\)]', key)]
            fraud_response[NON_FRAUD_EXPLANATION].append(non_fraud_text['Health Care Provider (Unknown)'])
            fraud_response[NON_FRAUD_EXPLANATION] = list(map(lambda st: str.replace(st, "Unknown", value[1]),
                                                             fraud_response[NON_FRAUD_EXPLANATION]))
        elif key.startswith('Attending'):
            value = [part.strip() for part in re.split('[\(\)]', key)]
            fraud_response[NON_FRAUD_EXPLANATION].append(non_fraud_text['Attending Physician (Unknown)'])
            fraud_response[NON_FRAUD_EXPLANATION] = list(map(lambda st: str.replace(st, "Unknown", value[1]),
                                                             fraud_response[NON_FRAUD_EXPLANATION]))
        elif key.startswith('Operating'):
            value = [part.strip() for part in re.split('[\(\)]', key)]
            fraud_response[NON_FRAUD_EXPLANATION].append(non_fraud_text['Operating Physician (Unknown)'])
            fraud_response[NON_FRAUD_EXPLANATION] = list(map(lambda st: str.replace(st, "Unknown", value[1]),
                                                             fraud_response[NON_FRAUD_EXPLANATION]))
        else:
            value = [part.strip() for part in re.split('[\(\)]', key)]
            range_min = int(float(value[1]) / num_dict[value[0]]) * num_dict[value[0]]
            range_max = range_min + num_dict[value[0]]
            json_key = f"{value[0]} ({range_min}-{range_max - 1})"
            if json_key in fraud_text.keys():
                fraud_response[NON_FRAUD_EXPLANATION].append(non_fraud_text[json_key])

    return fraud_response


# Function that takes 3 parameters to create a response of causality, fraud score and fraud status
def write_response(model_, single_data, feature_importance_dic):
    prediction_probabilities = model_.predict_proba(single_data)
    causality = get_casuality_data(feature_importance_dic)
    fraudScore = round((prediction_probabilities[0][1]).astype('float64'), fraudScore_Number_of_Decimals)
    fraudStatus = "fraud" if fraudScore > prediction_threshold else "non_fraud"
    text_explanation = get_text_explanation(causality_dict=causality)
    response = {"causality": causality, "fraudScore": fraudScore, "fraudStatus": fraudStatus,
                "explanation": text_explanation}
    return response


if __name__ == '__main__':
    pass
