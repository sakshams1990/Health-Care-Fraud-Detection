import pickle
from configparser import ConfigParser, ExtendedInterpolation

import numpy as np
import pandas as pd
from json_flatten import flatten

import Causality
import Shap


# Read Config.ini file to extract all the objects under PREDICTION section in dictionary format
def get_prediction_values_from_configuration_file():
    config = ConfigParser()
    config._interpolation = ExtendedInterpolation()
    config.read('Config.ini')
    pred_dic_value = dict(config.items('PREDICTION'))
    return pred_dic_value


# Reading all the pickle files and input json from the input source
def initialize_prediction():  # To read all the pickle files from the path mentioned in the config.ini
    prediction_value_dict = get_prediction_values_from_configuration_file()

    with open(prediction_value_dict['attendphy_codes'], "rb") as attendphy:
        attendphysician = pickle.load(attendphy)

    with open(prediction_value_dict['operatephy_codes'], "rb") as operatephy:  # Operating Physician
        operatephysician = pickle.load(operatephy)

    with open(prediction_value_dict['provider_codes'], "rb") as provider:  # Health Care Provider Codes
        providercode = pickle.load(provider)

    with open(prediction_value_dict['state_codes'], "rb") as state:
        statecode = pickle.load(state)

    with open(prediction_value_dict['model_pickle'], 'rb') as model_file:
        xgb_final_model = pickle.load(model_file)

    with open(prediction_value_dict['model_columns'], 'rb') as columns_file:
        column_names = pickle.load(columns_file)

    return attendphysician, operatephysician, providercode, statecode, xgb_final_model, column_names


def preprocess_json(inp_json):
    attend_physician, operate_physician, provider_code, state_code, xgb_final, model_columns = initialize_prediction()
    # Creating a dataframe for the nested json fields under ChronicCond.
    chronic_nested_params = pd.json_normalize(inp_json["ChronicCond"])
    # Creating a dataframe for the non-nested json fields
    params = pd.json_normalize(inp_json)
    # Concatenating params and nested_params dataframe
    merge_data = pd.concat([params, chronic_nested_params], axis=1)
    # Dropping ChronicCond column as it contains the original nested fields
    merge_data = merge_data.drop('ChronicCond', axis=1)
    # Rearranging the columns as per recognized by the model for prediction
    merge_data = merge_data[model_columns]
    merge_data = merge_data.applymap(lambda x: 1 if x == 'True' else x)  # Chronic Diseases
    merge_data = merge_data.applymap(lambda x: 0 if x == 'False' else x)  # Chronic Diseases
    merge_data = merge_data.applymap(lambda x: 1 if x == 'I' else x)  # Admission Type
    merge_data = merge_data.applymap(lambda x: 0 if x == 'O' else x)  # Admission Type

    # Converting Health Care , Attending Physician , Operating Physician in Upper Case
    merge_data['Health Care Provider'] = merge_data['Health Care Provider'].str.upper()
    merge_data['Attending Physician'] = merge_data['Attending Physician'].str.upper()
    merge_data['Operating Physician'] = merge_data['Operating Physician'].str.upper()

    # If the received provider code is present in provider code file, take the count frequency else consider as 1
    value_provider = inp_json['Health Care Provider']
    if value_provider in provider_code:
        merge_data['Health Care Provider'] = provider_code[value_provider]
    else:
        merge_data['Health Care Provider'] = 1
    # If the received attending physician is present in attending physician file, consider the count frequency else
    # consider value as 1
    value_attendphy = inp_json['Attending Physician']
    if value_attendphy in attend_physician:
        merge_data['Attending Physician'] = attend_physician[value_attendphy]
    else:
        merge_data['Attending Physician'] = 1
    # If the operating physician is present in the file, consider the count frequency else consider value as 1
    value_operatephy = inp_json['Operating Physician']
    if value_operatephy in operate_physician:
        merge_data['Operating Physician'] = operate_physician[value_operatephy]
    else:
        merge_data['Operating Physician'] = 1
    # If the state code is present in the file, consider the count frequency else consider the value as 1
    value_state = inp_json['State']
    if value_state in state_code:
        merge_data['State'] = state_code[value_state]
    else:
        merge_data['State'] = 1
    # If Admission Type is I, set OP Sum Assured and OP annual Premium as 0.0
    if inp_json['Admission Type'] == 'I':
        merge_data['OP Sum Assured'] = 0.0
        merge_data['OP Annual Premium'] = 0.0
    # If Admission Type is O, set IP Sum Assured and IP Annual Premium as 0.0
    else:
        merge_data['IP Sum Assured'] = 0.0
        merge_data['IP Annual Premium'] = 0.0
    # Creating an array that would be fed to model for prediction
    single_data_arr = np.array(merge_data.values.tolist())
    # Returning dataframe and list
    return merge_data, single_data_arr


# Converting the shap values into a list format
def shap_value_to_list(arr):
    arr = np.array(arr).tolist()[0]
    return arr


# Creating dictionary of the shap keys and values
def final_shap(key, value):
    shapDict = dict(zip(key, value))
    return shapDict


# Combine the keys and values of the input data and consider the combined as keys
def combine_input_keys_values(key, value):
    dic = {}
    combined_keys = [str(i) + ' (' + str(j) + ')' for i, j in zip(key, value)]
    for k in combined_keys:
        for val in value:
            dic[k] = val
            value.remove(val)
            break
    return dic


# Removing $ sign and ChronicCond.0. from the flatten data keys
def remove_chroniccond_dollar(key_list):
    for i in range(len(key_list)):  # Removing $ sign and ChronicCond.0 from the keys of the list
        key_list[i] = key_list[i].rsplit('$', 1)[0]  # Remove $ sign
        if key_list[i].startswith("ChronicCond.0."):  # Identifying elements starting with ChronicCond.0.
            key_list[i] = key_list[i][len("ChronicCond.0."):]  # Considering the element after remove of substring
    return key_list


# Creating a function that would call preprocess_json,shap_value_to_list,final_shap, write_response to create final
# response
def predict_single_json_input(data):
    attend_physician, operate_physician, provider_code, state_code, xgb_final, model_columns = initialize_prediction()
    df, single_data = preprocess_json(data)
    shap_values = shap_value_to_list(Shap.shap_value_xgboost_1_0_0(xgb_final, df))
    flat_data = flatten(data)  # Flattening the nested JSON using json_flatten package
    flat_data_keys = list(flat_data.keys())  # Converting the keys of dictionary into a list
    flat_data_keys = remove_chroniccond_dollar(flat_data_keys)
    final_data = dict(zip(flat_data_keys, flat_data.values()))
    final_dict = combine_input_keys_values(list(final_data.keys()), list(final_data.values()))
    shap_dict = final_shap(final_dict.keys(), shap_values)
    response = Causality.write_response(xgb_final, single_data, shap_dict)
    return response


if __name__ == '__main__':
    pass
