import shap

'''
Shap 0.35 works perfectly with xgboost version 1.0.0 . Apart from this combination, shap would throw an error. 
'''


def shap_value_xgboost_1_0_0(model, single_value_dataframe):
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    val = explainer.shap_values(single_value_dataframe)
    return val


if __name__ == '__main__':
    pass
