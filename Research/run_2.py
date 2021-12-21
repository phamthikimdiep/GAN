import math
import pickle
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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
        # data['time'] = data['time'].apply(lambda sec: (sec / 3600))
        # calculating hour of the day
        # data['cat_Hour'] = data['time'] % 24  # 2 days of data
        # data['cat_Hour'] = data['cat_Hour'].apply(lambda x: math.floor(x))
        # calculating 1st and 2nd day
        # data['cat_Day'] = data['time'] / 24  # 2 days of data
        # data['cat_Day'] = data['cat_Day'].apply(lambda x: 1 if (x == 0) else math.ceil(x))


    elif dataset_name == 'WA_Fn-UseC_-Telco-Customer-Churn':

        # data = data.drop('customerID', axis=1)
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

    df_metrics = pd.DataFrame()
    results_df = pd.DataFrame()
    df_metrics_sm = pd.DataFrame()
    results_df_sm = pd.DataFrame()
    df_mean_scores = pd.DataFrame()
    df_mean_scores_sm = pd.DataFrame()


    # make train-test split
    cat_cols = [col for col in data.columns if col.startswith("cat")]
    print("\n cat_cols: ",cat_cols)
    target = ['target']
    cols = [i for i in data.columns if i not in target]

    X = data[cols]
    Y = data[target]

    #X_train, X_test, Y_train, Y_test = train_test_split(
    #    X, Y,
    #    test_size=0.25,
    #    shuffle=False,
    #    random_state=42,
    #)

    print("======= X shape: {}  -  Y shape: {}\n".format(X.shape, Y.shape))


    #X_test, Y_test = X_test.reset_index(drop=True), Y_test.reset_index(drop=True)

    # oversampling minority class using smote
    os = SMOTE(random_state=0)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    id_fold = 0
    for train_ix, test_ix in kfold.split(X, Y):

        X_train_fold = X.iloc[train_ix]
        Y_train_fold = Y.iloc[train_ix]
        X_test_fold = X.iloc[test_ix]
        Y_test_fold = Y.iloc[test_ix]

        id_fold += 1

        print('-' * 30)
        print('          Fold - {}          '.format(id_fold))
        print('-' * 30)

        print("Shape X train fold: {} - Y train fold: {}\n".format(X_train_fold.shape,Y_train_fold.shape))

        iso = IsolationForest(contamination=0.011)
        yhat = iso.fit_predict(X_train_fold)

        mask = yhat != -1

        x_train, y_train = X_train_fold.iloc[mask, :], Y_train_fold.iloc[mask]

        #### smote
        print('=' * 10 + '> resampling - SMOTE')
        os_smote_x, os_smote_y = os.fit_resample(X_train_fold, Y_train_fold)
        x_sm_fold = pd.DataFrame(data=os_smote_x, columns=cols)
        y_sm_fold = pd.DataFrame(data=os_smote_y, columns=target)

        print("Shape X train smote: {} - Y train smote: {}\n".format(x_sm_fold.shape, y_sm_fold.shape))

        kinds = ['Gan','Smote']

        for kind in kinds:
            if kind == 'Gan':
                print("//" * 20)
                print("\nKind: ", kind)
                print("//" * 20)

                for train_prop_size in [0.05, 0.1, 0.25, 0.5, 0.75]:

                    #split
                    X_train_g, X_test_g, Y_train_g, Y_test_g = train_test_split(
                        X_train_fold,
                        Y_train_fold,
                        test_size=0.6,
                        shuffle=False,
                        random_state=42,
                    )

                    X_test_g, Y_test_g = X_test_g.reset_index(drop=True), Y_test_g.reset_index(drop=True)

                    train_size = X_train_g.shape[0]
                    print("==== train_prop_size: {}\n".format(train_prop_size))
                    print("==== train_size: {}\n".format(train_size))

                    X_train_g = X_train_g.head(int(train_size * train_prop_size)).reset_index(drop=True)
                    Y_train_g = Y_train_g.head(int(train_size * train_prop_size)).reset_index(drop=True)


                    print("==> X_train_g: {}  -  Y_train_g: {}\n".format(X_train_g.shape,Y_train_g.shape))
                    print("==> X_test_g: {}  -  Y_test_g: {}\n".format(X_test_g.shape, Y_test_g.shape))

                    mean_target_before_sampling_train = np.mean(Y_train_g)
                    if train_prop_size == 1:
                        continue
                    elif sample_type == "gan":
                        print("=== extend_gan_train ===")
                        X_train_g, Y_train_g = extend_gan_train(
                            X_train_g,
                            Y_train_g,
                            X_test_g,
                            cat_cols,
                            epochs=500,
                            gen_x_times=train_prop_size,
                        )
                    #elif sample_type == "sample_original":
                    #    print("=== extend_from_original ===")
                    #    X_train_g, Y_train_g = extend_from_original(
                    #        X_train_g, Y_train_g, X_test_g, cat_cols, gen_x_times=train_prop_size
                    #    )
                    Y_train_g, Y_test_g = Y_train_g, Y_test_g

                    for encoders_tuple in encoders_list:
                        for encoders_tuple in encoders_list:
                            print(
                                f"\n{encoders_tuple}, {dataset_name}, train size {int(100 * train_prop_size)}%, "
                                f"validation_type {validation_type}, sample_type {sample_type}"
                            )

                            time_start = time.time()

                            # train models
                            _model = Model(
                                cat_validation=validation_type,
                                encoders_names=encoders_tuple,
                                cat_cols=cat_cols,

                            )
                            # train_score, val_score, avg_num_trees, model = lgb_model.fit(X_train, y_train)
                            train_score, val_score, model = _model.fit(X_train_g, Y_train_g)
                            y_hat, test_features = _model.predict(X_test_g)
                            prediction = model.predict(X_test_g)

                            print("y test: {}\ny pred: {}\n".format(Y_test_g, prediction))

                            # save model
                            out_dir = f"./model/model_" + dataset_name + ".pkl"
                            pickle.dump(model, open(out_dir, 'wb'))

                            # check score
                            test_score = np.round(roc_auc_score(Y_test_g, y_hat), 4)
                            time_end = time.time()

                            # score
                            df_score = pd.DataFrame({
                                "dataset_name": dataset_name,
                                "metrics": "AUC",
                                "style": sample_type,
                                "Encoder": encoders_tuple[0],
                                "train_shape": X_train_g.shape[0],
                                "Train score": train_score,
                                "val_score": val_score}, index=[0])
                            # "avg_num_trees": avg_num_trees
                            df_scores = pd.DataFrame()
                            df_scores = df_scores.append(df_score, ignore_index=True)

                            # metrics

                            df_metric = model_metrics(Y_test_g, prediction, dataset_name, encoders_tuple[0],
                                                      train_prop_size,
                                                      X_train_g.shape[0],
                                                      validation_type, sample_type, id_fold)
                            df_metrics = df_metrics.append(df_metric, ignore_index=True)


                            # write and save results
                            results = {
                                "dataset_name": dataset_name,
                                "Encoder": encoders_tuple[0],
                                "validation_type": validation_type,
                                "sample_type": sample_type,
                                "train_shape": X_train_g.shape[0],
                                "test_shape": X_test_g.shape[0],
                                "mean_target_before_sampling_train": mean_target_before_sampling_train,
                                "mean_target_after_sampling_train": np.round(np.mean(Y_train_g), 4),
                                "mean_target_test": np.round(np.mean(Y_test_g), 4),
                                "num_cat_cols": len(cat_cols),
                                "train_score": train_score,
                                "val_score": val_score,
                                "test_score": test_score,
                                "time": np.round(time_end - time_start, 4),
                                "features_before_encoding": X_train_g.shape[1],
                                "features_after_encoding": test_features,
                                # "avg_tress_number": avg_num_trees,
                                "train_prop_size": train_prop_size,
                                "Fold": id_fold,
                            }
                            save_exp_to_file(dic=results, path="./results/fit_predict_scores.txt")
                            results_df = results_df.append([results], ignore_index=True)

                        #writer = pd.ExcelWriter(
                        #    "./results/fit_predict_scores_" + dataset_name + "_" + str(sample_type) + ".xlsx")
                        #results_df.to_excel(writer, index=False)
                        #writer.save()

                        writer_metric = pd.ExcelWriter(
                            "./results/metrics_" + dataset_name + "_" + str(sample_type) + ".xlsx")
                        df_metrics.to_excel(writer_metric, index=False)
                        writer_metric.save()



                        #writer_score = pd.ExcelWriter(
                        #    "./results/train_scores_" + dataset_name + "_" + str(sample_type) + ".xlsx")
                        #df_scores.to_excel(writer_score, index=False)
                        #writer_score.save()

            elif kind == 'Smote':
                print("//" * 20)
                print("\nKind: ", kind)
                print("//" * 20)

                for train_prop_size in [0.05, 0.1, 0.25, 0.5, 0.75]:

                    # split
                    X_train_gsm, X_test_gsm, Y_train_gsm, Y_test_gsm = train_test_split(
                        x_sm_fold,
                        y_sm_fold,
                        test_size=0.6,
                        shuffle=False,
                        random_state=42,
                    )

                    X_test_gsm, Y_test_gsm = X_test_gsm.reset_index(drop=True), Y_test_gsm.reset_index(drop=True)

                    train_sizesm = X_train_gsm.shape[0]
                    print("train_prop_size: {}\n".format(train_prop_size))
                    print("train_size: {}\n".format(train_sizesm))

                    X_train_gsm = X_train_gsm.head(int(train_sizesm * train_prop_size)).reset_index(drop=True)
                    Y_train_gsm = Y_train_gsm.head(int(train_sizesm * train_prop_size)).reset_index(drop=True)

                    print("==> X_train_sm: {}  -  Y_train_sm: {}\n".format(X_train_gsm.shape, Y_train_gsm.shape))
                    print("==> X_testsm: {}  -  Y_testsm: {}\n".format(X_test_gsm.shape, Y_test_gsm.shape))

                    mean_target_before_sampling_train_sm = np.mean(Y_train_gsm)
                    if train_prop_size == 1:
                        continue
                    elif sample_type == "gan":
                        print("=== extend_gan_train_smote ===")
                        X_train_gsm, Y_train_gsm = extend_gan_train(
                            X_train_gsm,
                            Y_train_gsm,
                            X_test_gsm,
                            cat_cols,
                            epochs=500,
                            gen_x_times=train_prop_size,
                        )
                    #elif sample_type == "sample_original":
                    #    print("=== sample_original_smote ===")
                    #    X_train_gsm, Y_train_gsm = extend_from_original(
                    #        X_train_gsm, Y_train_gsm, X_test_gsm, cat_cols, gen_x_times=train_prop_size
                    #    )
                    Y_train_gsm, Y_test_gsm = Y_train_gsm, Y_test_gsm

                    for encoders_tuple in encoders_list:
                        for encoders_tuple in encoders_list:
                            print(
                                f"\n{encoders_tuple}, {dataset_name}, train size {int(100 * train_prop_size)}%, "
                                f"validation_type {validation_type}, sample_type {sample_type}"
                            )

                            time_start = time.time()

                            # train models
                            _model_sm = Model(
                                cat_validation=validation_type,
                                encoders_names=encoders_tuple,
                                cat_cols=cat_cols,

                            )
                            # train_score, val_score, avg_num_trees, model = lgb_model.fit(X_train, y_train)
                            train_score_sm, val_score_sm, model_sm = _model_sm.fit(X_train_gsm, Y_train_gsm)
                            y_hat_sm, test_features_sm = _model_sm.predict(X_test_gsm)
                            prediction_sm = model_sm.predict(X_test_gsm)


                            print("y test: {}\ny pred: {}\n".format(Y_test_gsm,prediction_sm))

                            # save model
                            out_dir_sm = f"./model/model_" + dataset_name + "_sm.pkl"
                            pickle.dump(model_sm, open(out_dir_sm, 'wb'))

                            # check score
                            test_score_sm = np.round(roc_auc_score(Y_test_gsm, y_hat_sm), 4)
                            time_end = time.time()

                            # score
                            df_score_sm = pd.DataFrame({
                                "dataset_name": dataset_name,
                                "metrics": "AUC",
                                "style": sample_type,
                                "Encoder": encoders_tuple[0],
                                "train_shape": X_train_gsm.shape[0],
                                "Train score": train_score_sm,
                                "val_score": val_score_sm}, index=[0])
                            # "avg_num_trees": avg_num_trees
                            df_scores_sm = pd.DataFrame()
                            df_scores_sm = df_scores_sm.append(df_score_sm, ignore_index=True)

                            # metrics

                            df_metric_sm = model_metrics(Y_test_gsm, prediction_sm, dataset_name, encoders_tuple[0],
                                                      train_prop_size,
                                                      X_train_gsm.shape[0],
                                                      validation_type, sample_type, id_fold)

                            df_metrics_sm = df_metrics_sm.append(df_metric_sm, ignore_index=True)



                            # write and save results
                            results_sm = {
                                "dataset_name": dataset_name,
                                "Encoder": encoders_tuple[0],
                                "validation_type": validation_type,
                                "sample_type": sample_type,
                                "train_shape": X_train_gsm.shape[0],
                                "test_shape": X_test_gsm.shape[0],
                                "mean_target_before_sampling_train": mean_target_before_sampling_train_sm,
                                "mean_target_after_sampling_train": np.round(np.mean(Y_train_gsm), 4),
                                "mean_target_test": np.round(np.mean(Y_test_gsm), 4),
                                "num_cat_cols": len(cat_cols),
                                "train_score": train_score_sm,
                                "val_score": val_score_sm,
                                "test_score": test_score_sm,
                                "time": np.round(time_end - time_start, 4),
                                "features_before_encoding": X_train_gsm.shape[1],
                                "features_after_encoding": test_features_sm,
                                # "avg_tress_number": avg_num_trees,
                                "train_prop_size": train_prop_size,
                            }
                            save_exp_to_file(dic=results_sm, path="./results/fit_predict_scores_sm.txt")
                            results_df_sm = results_df_sm.append([results_sm], ignore_index=True)

                        # writer_sm = pd.ExcelWriter(
                        #    "./results/fit_predict_scores_" + dataset_name + "_" + str(sample_type) + "_sm.xlsx")
                        # results_df_sm.to_excel(writer_sm, index=False)
                        # writer_sm.save()

                        writer_metric_sm = pd.ExcelWriter(
                            "./results/metrics_" + dataset_name + "_" + str(sample_type) + "_sm.xlsx")
                        df_metrics_sm.to_excel(writer_metric_sm, index=False)
                        writer_metric_sm.save()



                        # writer_score_sm = pd.ExcelWriter(
                        #    "./results/train_scores_" + dataset_name + "_" + str(sample_type) + "_sm.xlsx")
                        # df_scores_sm.to_excel(writer_score_sm, index=False)
                        # writer_score_sm.save()




if __name__ == "__main__":

    # Other type of enccoders might be used as well


    #encoders_list = [("FrequencyEncoder",)]
    #encoders_list = [("WOEEncoder",)]
    #encoders_list = [("TargetEncoder",)]
    #encoders_list = [("SumEncoder",)]
    #encoders_list = [("MEstimateEncoder",)]
    #encoders_list = [("LeaveOneOutEncoder",)]
    #encoders_list = [("HelmertEncoder",)]
    #encoders_list = [("BackwardDifferenceEncoder",)]
    #encoders_list = [("JamesSteinEncoder",)]

    encoders_list = [("OrdinalEncoder",)]
    #encoders_list = [("CatBoostEncoder",)]


    dataset_list = [
        #"BankruptcyPrediction",
        #"online_shoppers_intention",
        "creditcard",
        #"WA_Fn-UseC_-Telco-Customer-Churn",
    ]

    for dataset_name in tqdm(dataset_list):
        print('\n', '-' * 20, dataset_name, '-' * 20, '\n')
        validation_type = "Single"
        print("-"*20)
        print('\n******** sample_type --- None', '\n')
        print("-" * 20)
        #execute_experiment(dataset_name, encoders_list, validation_type)
        print("-" * 20)
        print('\n******** sample_type --- gan', '\n')
        print("-" * 20)
        execute_experiment(dataset_name, encoders_list, validation_type, sample_type="gan")
        print("-" * 20)
        print('\n******** sample_type --- sample_original', '\n')
        print("-" * 20)
        #execute_experiment(dataset_name, encoders_list, validation_type, sample_type="sample_original")