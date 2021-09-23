import math
import pickle
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import Model
from utils import *
from pandas import ExcelWriter
from Functions import *

import warnings

warnings.filterwarnings("ignore")


# remove special characters
# remove(filename, '\/:*?"<>|')
def remove(value, deletechars):
    for c in deletechars:
        value = value.replace(c, '')
    return value


def convert(data):
    number = preprocessing.LabelEncoder()
    data['cat_Month'] = number.fit_transform(data['cat_Month'])
    data['cat_VisitorType'] = number.fit_transform(data['cat_VisitorType'])
    data = data.fillna(-9999)
    return data


def execute_experiment(dataset_name, encoders_list, validation_type, sample_type=None):
    """
    Executes experiment with specified dataset name, sample strategy and validation type. Metrics and results will written in log file
    Args:
        dataset_name: dataset name, which will read from data folder as csv file
        encoders_list: encoders_list which will be used for training
        validation_type: categorical type of validation, examples: "None", "Single" and "Double"
        sample_type: sample type by generating from gan or by sampling from train
    Returns: None
    """
    dataset_pth = f"./data/{dataset_name}.csv"
    results = {}
    remove(dataset_pth, '\/:*?"<>|')

    # load processed dataset
    data = pd.read_csv(dataset_pth)
    data = pd.DataFrame(data)
    data.fillna(data.mean(), inplace=True)

    if dataset_name == 'online_shoppers_intention':
        data = convert(data)
    elif dataset_name == "creditcard":
        total = data.isnull().sum().sort_values(ascending=False)
        percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
        pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
        data.fillna(0, inplace=True)
        # converting time from second to hour
        data['time'] = data['time'].apply(lambda sec: (sec / 3600))
        # calculating hour of the day
        data['cat_Hour'] = data['time'] % 24  # 2 days of data
        data['cat_Hour'] = data['cat_Hour'].apply(lambda x: math.floor(x))
        # calculating 1st and 2nd day
        data['cat_Day'] = data['time'] / 24  # 2 days of data
        data['cat_Day'] = data['cat_Day'].apply(lambda x: 1 if (x == 0) else math.ceil(x))


    elif dataset_name == 'WA_Fn-UseC_-Telco-Customer-Churn':
        data = data.drop('customerID', axis=1)
        # 1- Replacing spaces with null values in Total charges column
        data['num_TotalCharges'] = data['num_TotalCharges'].replace(" ", np.nan)
        # 2 - Dropping null values from Total charges column which contain .15% missing data
        data = data[data['num_TotalCharges'].notnull()]
        data = data.reset_index()[data.columns]
        # 3 - Convert to float type
        data["num_TotalCharges"] = data["num_TotalCharges"].astype(float)
        # 4 - Replace 'No internet service' to No for the following columns
        replace_cols = ['cat_OnlineSecurity',
                        'cat_OnlineBackup',
                        'cat_DeviceProtection',
                        'cat_TechSupport',
                        'cat_StreamingTV',
                        'cat_StreamingMovies']

        for i in replace_cols:
            data[i].replace({'No internet service': 'No'}, inplace=True)

        # 5 - Replace values
        data['cat_gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
        data['cat_MultipleLines'].replace({'No phone service': 'No'}, inplace=True)
        data['cat_InternetService'].replace({'DSL': 'Yes', 'Fiber optic': 'Yes'}, inplace=True)
        data['cat_Contract'].replace({'Month-to-month': 1, 'One year': 2, 'Two year': 3}, inplace=True)
        data['cat_PaymentMethod'].replace({'Electronic check': 1,
                                           'Mailed check': 2,
                                           'Bank transfer (automatic)': 3,
                                           'Credit card (automatic)': 4}, inplace=True)

        replace_col2 = ['cat_Partner',
                        'cat_Dependents',
                        'cat_PhoneService',
                        'cat_MultipleLines',
                        'cat_OnlineSecurity',
                        'cat_OnlineBackup',
                        'cat_DeviceProtection',
                        'cat_TechSupport',
                        'cat_StreamingTV',
                        'cat_StreamingMovies',
                        'cat_PaperlessBilling',
                        'cat_MultipleLines',
                        'cat_InternetService']
        for i in replace_col2:
            data[i].replace({"Yes": 1, "No": 0}, inplace=True)

        # 6 - Tenure to categorical column
        def tenure_lab(telcom):
            if telcom["tenure"] <= 12:
                return 0 #"Tenure_0-12"
            elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24):
                return 1 #"Tenure_12-24"
            elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48):
                return 2 #"Tenure_24-48"
            elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60):
                return 3 #"Tenure_48-60"
            elif telcom["tenure"] > 60:
                return 4 #"Tenure_gt_60"

        data["cat_tenure"] = data.apply(lambda data: tenure_lab(data), axis=1)

    df_metrics = pd.DataFrame()
    results_df = pd.DataFrame()
    df_metrics_sm = pd.DataFrame()
    results_df_sm = pd.DataFrame()

    for train_prop_size in [0.05, 0.1, 0.25, 0.5, 0.75]:
        # make train-test split
        cat_cols = [col for col in data.columns if col.startswith("cat")]
        target = ['target']
        cols = [i for i in data.columns if i not in target]

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop("target", axis=1),
            data["target"],
            test_size=0.6,
            shuffle=False,
            random_state=42,
        )

        # oversampling minority class using smote

        os = SMOTE(random_state=0)
        os_smote_x, os_smote_y = os.fit_resample(X_train, y_train)
        X_train_sm = pd.DataFrame(data=os_smote_x, columns=cols)
        y_train_sm = pd.DataFrame(data=os_smote_y, columns=target)

        X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

        train_size = X_train.shape[0]
        X_train = X_train.head(int(train_size * train_prop_size)).reset_index(drop=True)
        y_train = y_train.head(int(train_size * train_prop_size)).reset_index(drop=True)

        kinds = ['None', 'Smote']
        for kind in kinds:
            if kind == 'None':
                print("//" * 20)
                print("\nKind: ", kind)
                print("//" * 20)
                mean_target_before_sampling_train = np.mean(y_train)
                if train_prop_size == 1:
                    continue
                elif sample_type == "gan":
                    print("//" * 20,"Gan_None")
                    print('X train shape: ',X_train.shape)
                    print('Y train shape: ',y_train.shape)
                    '''
                    X_train, y_train = extend_gan_train(
                        X_train,
                        y_train,
                        X_test,
                        cat_cols,
                        epochs=500,
                        gen_x_times=train_prop_size
                    )
                    '''

                elif sample_type == "sample_original":
                    print("//" * 20, "sample_original_None")
                    print('X train shape: ', X_train.shape)
                    print('Y train shape: ', y_train.shape)
                    '''
                    X_train, y_train = extend_from_original(
                        X_train, y_train, X_test, cat_cols, gen_x_times=train_prop_size
                    )
                    '''
                y_train, y_test = y_train, y_test

                for encoders_tuple in encoders_list:
                    print(
                        f"\n{encoders_tuple}, {dataset_name}, train size {int(100 * train_prop_size)}%, "
                        f"validation_type {validation_type}, sample_type {sample_type}"
                    )

                    time_start = time.time()

                    # train models
                    lgb_model = Model(
                        cat_validation=validation_type,
                        encoders_names=encoders_tuple,
                        cat_cols=cat_cols,
                    )
                    train_score, val_score, avg_num_trees, model = lgb_model.fit(X_train, y_train)
                    y_hat, test_features = lgb_model.predict(X_test)
                    prediction = model.predict(X_test)

                    # save model
                    out_dir = f"./model/model_" + dataset_name + ".pkl"
                    pickle.dump(model, open(out_dir, 'wb'))

                    # check score
                    test_score = roc_auc_score(y_test, y_hat)
                    time_end = time.time()

                    # metrics

                    df_metric = model_metrics(y_test, prediction, dataset_name, encoders_tuple[0], X_train.shape[0],
                                              validation_type, sample_type)
                    df_metrics = df_metrics.append(df_metric, ignore_index=True)

                    # write and save results
                    results = {
                        "dataset_name": dataset_name,
                        "Encoder": encoders_tuple[0],
                        "validation_type": validation_type,
                        "sample_type": sample_type,
                        "train_shape": X_train.shape[0],
                        "test_shape": X_test.shape[0],
                        "mean_target_before_sampling_train": mean_target_before_sampling_train,
                        "mean_target_after_sampling_train": np.mean(y_train),
                        "mean_target_test": np.mean(y_test),
                        "num_cat_cols": len(cat_cols),
                        "train_score": train_score,
                        "val_score": val_score,
                        "test_score": test_score,
                        "time": time_end - time_start,
                        "features_before_encoding": X_train.shape[1],
                        "features_after_encoding": test_features,
                        "avg_tress_number": avg_num_trees,
                        "train_prop_size": train_prop_size,
                    }
                    save_exp_to_file(dic=results, path="./results/fit_predict_scores.txt")
                    results_df = results_df.append([results], ignore_index=True)

                writer = pd.ExcelWriter(
                    "./results/fit_predict_scores_" + dataset_name + "_" + str(sample_type) + ".xlsx")
                results_df.to_excel(writer, index=False)
                writer.save()

                writer_metric = pd.ExcelWriter("./results/metrics_" + dataset_name + "_" + str(sample_type) + ".xlsx")
                df_metrics.to_excel(writer_metric, index=False)
                writer_metric.save()
            elif kind == 'Smote':
                print("//" * 20)
                print("\nKind: ", kind)
                print("//" * 20)


                mean_target_before_sampling_train_sm = np.mean(y_train_sm)
                if train_prop_size == 1:
                    continue
                elif sample_type == "gan":
                    print("\ngan: \n","X_train_sm: ", X_train_sm.shape, " - y_train_sm: ",y_train_sm.shape)

                    '''
                    X_train_sm, y_train_sm = extend_gan_train(
                        X_train_sm,
                        y_train_sm,
                        X_test,
                        cat_cols,
                        epochs=500,
                        gen_x_times=train_prop_size
                    )
                    '''


                elif sample_type == "sample_original":
                    print("\nsample_original: \n", "X_train_sm: ", X_train_sm.shape, " - y_train_sm: ", y_train_sm.shape)
                    '''
                    X_train_sm, y_train_sm = extend_from_original(
                        X_train_sm, y_train_sm, X_test, cat_cols, gen_x_times=train_prop_size
                    )
                    '''

                y_train_sm, y_test = y_train_sm, y_test

                for encoders_tuple in encoders_list:
                    print(
                        f"\n{encoders_tuple}, {dataset_name}, train size {int(100 * train_prop_size)}%, "
                        f"validation_type {validation_type}, sample_type {sample_type}"
                    )

                    time_start_sm = time.time()

                    # train models
                    lgb_model_sm = Model(
                        cat_validation=validation_type,
                        encoders_names=encoders_tuple,
                        cat_cols=cat_cols,
                    )
                    train_score_sm, val_score_sm, avg_num_trees_sm, model_sm = lgb_model_sm.fit(X_train_sm, y_train_sm)
                    y_hat_sm, test_features_sm = lgb_model_sm.predict(X_test)
                    prediction_sm = model_sm.predict(X_test)

                    # save model
                    out_dir = f"./model/model_" + dataset_name + "_sm.pkl"
                    pickle.dump(model_sm, open(out_dir, 'wb'))

                    # check score
                    test_score_sm = roc_auc_score(y_test, y_hat_sm)
                    time_end_sm = time.time()

                    # metrics

                    df_metric_sm = model_metrics(y_test, prediction_sm, dataset_name, encoders_tuple[0],
                                                 X_train_sm.shape[0],
                                                 validation_type, sample_type)
                    df_metrics_sm = df_metrics_sm.append(df_metric_sm, ignore_index=True)

                    # write and save results
                    results_sm = {
                        "dataset_name": dataset_name,
                        "Encoder": encoders_tuple[0],
                        "validation_type": validation_type,
                        "sample_type": sample_type,
                        "train_shape": X_train_sm.shape[0],
                        "test_shape": X_test.shape[0],
                        "mean_target_before_sampling_train": mean_target_before_sampling_train_sm,
                        "mean_target_after_sampling_train": np.mean(y_train_sm),
                        "mean_target_test": np.mean(y_test),
                        "num_cat_cols": len(cat_cols),
                        "train_score": train_score_sm,
                        "val_score": val_score_sm,
                        "test_score": test_score_sm,
                        "time": time_end_sm - time_start_sm,
                        "features_before_encoding": X_train_sm.shape[1],
                        "features_after_encoding": test_features_sm,
                        "avg_tress_number": avg_num_trees_sm,
                        "train_prop_size": train_prop_size,
                    }
                    save_exp_to_file(dic=results_sm, path="./results/fit_predict_scores_sm.txt")
                    results_df_sm = results_df_sm.append([results_sm], ignore_index=True)

                writer = pd.ExcelWriter(
                    "./results/fit_predict_scores_" + dataset_name + "_" + str(sample_type) + "_sm.xlsx")
                results_df.to_excel(writer, index=False)
                writer.save()

                writer_metric = pd.ExcelWriter(
                    "./results/metrics_" + dataset_name + "_" + str(sample_type) + "_sm.xlsx")
                df_metrics.to_excel(writer_metric, index=False)
                writer_metric.save()

                print("Train size:", train_prop_size)
                print("X_train_sm:", X_train_sm.shape)
                print("y_train_sm:", y_train_sm.shape)


if __name__ == "__main__":

    # Other type of enccoders might be used as well
    encoders_list = [("CatBoostEncoder",)]

    dataset_list = [
        "BankruptcyPrediction",
        "creditcard",
        "online_shoppers_intention",
        "WA_Fn-UseC_-Telco-Customer-Churn",
    ]

    for dataset_name in tqdm(dataset_list):
        print('-' * 20, dataset_name, '-' * 20, '\n')
        validation_type = "Single"
        print("//"*20)
        print('******** sample_type --- None', '\n')
        print("//" * 20)
        execute_experiment(dataset_name, encoders_list, validation_type)
        print("//" * 20)
        print('******** sample_type --- gan', '\n')
        print("//" * 20)
        execute_experiment(dataset_name, encoders_list, validation_type, sample_type="gan")
        print("//" * 20)
        print('******** sample_type --- sample_original', '\n')
        print("//" * 20)
        execute_experiment(dataset_name, encoders_list, validation_type, sample_type="sample_original")
