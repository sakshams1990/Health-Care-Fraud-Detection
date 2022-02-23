import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from sklearn.metrics import roc_curve, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split

import Utils_Configurations


# Get Model Configurations from Config.ini file
def get_model_configuration(config_file_name, config_section):
    config_file_path = os.path.join(os.getcwd(), config_file_name)
    config = Utils_Configurations.Configuration(config_file_path)
    no_of_splits = config.read_configuration_options(config_section, "n_splits", "int")
    split_size = config.read_configuration_options(config_section, "test_size", "float")
    random_number = config.read_configuration_options(config_section, "random_state", "int")
    shuffle_data = config.read_configuration_options(config_section, "shuffle", "bool")
    return no_of_splits, split_size, random_number, shuffle_data


# Read the dataset and the datatype pickle file
def read_data(datatypefile, data_file):
    with open(datatypefile, 'rb') as dt:
        datatype = pickle.load(dt)
    df_data_file = pd.read_csv(data_file, dtype=datatype)
    return df_data_file


# Split X & Y
def split_x_y(df, label_name):
    # Split X and y
    x_fraud = df.drop([label_name], axis=1)
    y_fraud = df[label_name]
    return x_fraud, y_fraud


# Train-Test split
def split_train_test(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x.values, y.values, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    return train_x, test_x, train_y, test_y


# Function for building model and predicting on X_test
def model_building_predictions(train_x, test_x, train_y, test_y):
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "max_depth": [3, 4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}
    strkf = StratifiedKFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    # XGBoost Classifier
    xgb_class = xgb.XGBClassifier()
    random_xgb = RandomizedSearchCV(xgb_class, param_distributions=param_grid,
                                    cv=strkf, random_state=random_state)
    random_xgb.fit(train_x, train_y)
    model_param_xg = random_xgb.best_params_

    # Printing the best score and the best parameters found by RandomizedSearchCV
    print("Best: %f using %s" % (random_xgb.best_score_, random_xgb.best_params_))

    fraud_model = xgb.XGBClassifier(min_child_weight=model_param_xg['min_child_weight'],
                                    max_depth=model_param_xg['max_depth'],
                                    learning_rate=model_param_xg['learning_rate'],
                                    gamma=model_param_xg['gamma'], colsample_bytree=model_param_xg['colsample_bytree'],
                                    random_state=random_state)
    fraud_model.fit(train_x, train_y)
    pkl_xgb = "../Fraud Detection/Trained Model/xgb_pickle_model.pkl"
    with open(pkl_xgb, 'wb') as file:
        pickle.dump(fraud_model, file)

    xgb_pred = fraud_model.predict(test_x)
    xgb_pred_proba = fraud_model.predict_proba(test_x)[:, 1]
    print('AUC Score for XGBoost Classifier is:', roc_auc_score(test_y, xgb_pred))
    print('F1 Score for XGBoost Classifier is:', f1_score(test_y, xgb_pred))
    print('Recall score for XGBoost Classifier is:', recall_score(test_y, xgb_pred))
    print('Precision score for XGBoost Classifier is:', precision_score(test_y, xgb_pred))
    print('Accuracy score for XGBoost Classifier is:', accuracy_score(test_y, xgb_pred))
    return xgb_pred, xgb_pred_proba


# Confusion Metrics and
def model_visualisations(test_y, pred_y, pred_proba_y):
    xgb_roc_auc = roc_auc_score(test_y, pred_y)
    fpr, tpr, thresholds = roc_curve(test_y, pred_proba_y)
    plt.figure()
    plt.plot(fpr, tpr, label='XGB (area = %0.2f)' % xgb_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('../Fraud Detection/Model Visualization/AUC-ROC Curve.png')

    cnf_matrix_xg = confusion_matrix(test_y, pred_y)
    # Confusion Matrix for XGBoost Classifier model
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix_xg), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('../Fraud Detection/Model Visualization/Confusion Metrics.png')


if __name__ == '__main__':
    n_split, test_size, random_state, shuffle = get_model_configuration(config_file_name="Config.ini",
                                                                        config_section="MODEL CONFIG")
    df_data = read_data(datatypefile="../Fraud Detection/Dataset/Final/data_type_model.pkl",
                        data_file="../Fraud Detection/Dataset/Final/fraud_model.csv")
    X, y = split_x_y(df_data, label_name='Potential Fraud')
    X_train, X_test, y_train, y_test = split_train_test(x=X, y=y)
    y_pred, y_pred_proba = model_building_predictions(X_train, X_test, y_train, y_test)
    model_visualisations(y_test, y_pred, y_pred_proba)
