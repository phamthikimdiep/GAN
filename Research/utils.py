import gc
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ctgan import _CTGANSynthesizer
from model import Model
import warnings
warnings.filterwarnings("ignore")

# Metrics
from sklearn import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.metrics import geometric_mean_score

def save_dict_to_file(dic: dict, path: str, save_raw=False) -> None:
    """
    Save dict values into txt file
    :param dic: Dict with values
    :param path: Path to .txt file
    :return: None
    """

    f = open(path, "w")
    if save_raw:
        f.write(str(dic))
    else:
        for k, v in dic.items():
            f.write(str(k))
            f.write(str(v))
            f.write("\n\n")
    f.close()

def save_exp_to_file(dic: dict, path: str) -> None:
    """
    Save dict values into txt file
    :param dic: Dict with values
    :param path: Path to .txt file
    :return: None
    """

    f = open(path, "a+")
    keys = dic.keys()
    vals = [str(val) for val in dic.values()]

    if f.tell() == 0:
        header = "\t".join(keys)
        f.write(header + "\n")

    row = "\t".join(vals)
    f.write(row + "\n")
    f.close()

def cat_cols_info(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: List[str]
) -> dict:
    """
    Get the main info about cat columns in dataframe, i.e. num of values, uniqueness
    :param X_train: Train dataframe
    :param X_test: Test dataframe
    :param cat_cols: List of categorical columns
    :return: Dict with results
    """

    cc_info = {}

    for col in cat_cols:
        train_values = set(X_train[col])
        number_of_new_test = len(set(X_test[col]) - train_values)
        fraction_of_new_test = np.mean(
            X_test[col].apply(lambda v: v not in train_values)
        )

        cc_info[col] = {
            "num_uniq_train": X_train[col].nunique(),
            "num_uniq_test": X_test[col].nunique(),
            "number_of_new_test": number_of_new_test,
            "fraction_of_new_test": fraction_of_new_test,
        }
    return cc_info

def adversarial_test(left_df, right_df, cat_cols):
    """
    Trains adversarial model to distinguish train from test
    :param left_df:  dataframe
    :param right_df: dataframe
    :param cat_cols: List of categorical columns
    :return: trained model
    """
    # sample to shuffle the data
    left_df = left_df.copy().sample(frac=1).reset_index(drop=True)
    right_df = right_df.copy().sample(frac=1).reset_index(drop=True)

    left_df = left_df.head(right_df.shape[0])
    right_df = right_df.head(left_df.shape[0])

    left_df["gt"] = 0
    right_df["gt"] = 1

    concated = pd.concat([left_df, right_df])

    _model = Model(
        cat_validation="Single",
        encoders_names=("OrdinalEncoder",),
        cat_cols=cat_cols,
        model_validation=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        model_params={

            # lgbm params
            #"metrics": "AUC",
            #"max_bin":100,
            #"n_estimators": 100,
            #"learning_rate": 0.5,
            #"random_state": 42,
            #"max_depth": 2,


            # randomforest
            #"max_depth": 2,
            #"n_estimators": 100,
            #"random_state": 42,
            #"criterion": "gini",

            # xgb params
            "n_estimators": 100,
            "learning_rate": 0.5,
            "random_state": 42,
            "max_depth": 2,
            "eval_metric": "logloss",



        },
    )

    """
    train_score, val_score, avg_num_trees, model = _model.fit(
        concated.drop("gt", axis=1), concated["gt"]
    )
    """
    train_score, val_score,  model = _model.fit(
        concated.drop("gt", axis=1), concated["gt"]
    )

    print(
        "ROC AUC adversarial: train %.2f%% val %.2f%%"
        % (train_score * 100.0, val_score * 100.0)
    )
    return _model

def extend_gan_train(x_train, y_train, x_test, cat_cols, gen_x_times=1.2, epochs=300):
    """
    Extends train by generating new data by GAN
    :param x_train:  train dataframe
    :param y_train: target for train dataframe
    :param x_test: dataframe
    :param cat_cols: List of categorical columns
    :param gen_x_times: Factor for which initial dataframe should be increased
    :param cat_cols: List of categorical columns
    :param epochs: Number of epoch max to train the GAN
    :return: extended train with target
    """

    if gen_x_times == 0:
        raise ValueError("Passed gen_x_times with value 0!")
    x_train["target"] = y_train
    x_test_bigger = round(int(gen_x_times * x_test.shape[0] / x_train.shape[0]))
    print("x_test_bigger: ",x_test_bigger)
    ctgan = _CTGANSynthesizer()
    ctgan.fit(x_train, cat_cols, epochs=epochs)
    generated_df = ctgan.sample((x_test_bigger) * x_train.shape[0])
    data_dtype = x_train.dtypes.values

    for i in range(len(generated_df.columns)):
        generated_df[generated_df.columns[i]] = generated_df[
            generated_df.columns[i]
        ].astype(data_dtype[i])

    generated_df = pd.concat(
        [
            x_train,
            generated_df,
        ]
    ).reset_index(drop=True)

    num_cols = []
    for col in x_train.columns:
        if "num" in col:
            num_cols.append(col)

    for num_col in num_cols:
        min_val = x_test[num_col].quantile(0.02)
        max_val = x_test[num_col].quantile(0.98)
        generated_df = generated_df.loc[
            (generated_df[num_col] >= min_val) & (generated_df[num_col] <= max_val)
        ]
    generated_df = generated_df.reset_index(drop=True)
    ad_model = adversarial_test(x_test, generated_df.drop("target", axis=1), cat_cols)

    generated_df["test_similarity"] = ad_model.predict(
        generated_df.drop("target", axis=1), return_shape=False
    )
    generated_df.sort_values("test_similarity", ascending=False, inplace=True)
    generated_df = generated_df.head(int(gen_x_times * x_train.shape[0]))
    x_train = pd.concat(
        [x_train, generated_df.drop("test_similarity", axis=1)], axis=0
    ).reset_index(drop=True)

    del generated_df
    gc.collect()
    return x_train.drop("target", axis=1), x_train["target"]

def extend_from_original(x_train, y_train, x_test, cat_cols, gen_x_times=1.1):
    """
    Extends train by generating new data by GAN
    :param x_train:  train dataframe
    :param y_train: target for train dataframe
    :param x_test: test dataframe
    :param cat_cols: List of categorical columns
    :param gen_x_times: Factor for which initial dataframe should be increased
    :return: extended train with target
    """
    if gen_x_times == 0:
        raise ValueError("Passed gen_x_times with value 0!")
    x_train["target"] = y_train
    x_test_bigger = round(int(gen_x_times * x_test.shape[0] / x_train.shape[0]))
    generated_df = x_train.sample(frac=x_test_bigger, replace=True, random_state=42)
    num_cols = []
    for col in x_train.columns:
        if "num" in col:
            num_cols.append(col)

    for num_col in num_cols:
        min_val = x_test[num_col].quantile(0.02)
        max_val = x_test[num_col].quantile(0.98)
        generated_df = generated_df.loc[
            (generated_df[num_col] >= min_val) & (generated_df[num_col] <= max_val)
        ]

    generated_df = generated_df.reset_index(drop=True)
    ad_model = adversarial_test(x_test, generated_df.drop("target", axis=1), cat_cols)

    generated_df["test_similarity"] = ad_model.predict(
        generated_df.drop("target", axis=1), return_shape=False
    )
    generated_df.sort_values("test_similarity", ascending=False, inplace=True)
    generated_df = generated_df.head(int(gen_x_times * x_train.shape[0]))
    x_train = pd.concat(
        [x_train, generated_df.drop("test_similarity", axis=1)], axis=0
    ).reset_index(drop=True)
    del generated_df
    gc.collect()
    return x_train.drop("target", axis=1), x_train["target"]

scores_list_acc = []
scores_list_recall = []
scores_list_precision = []
scores_list_f1 = []
scores_list_auc = []
scores_list_kappa = []
scores_list_gmean = []

def model_metrics(y_test, prediction,dataset_name,encoder,trainsize,trainshape,validation_type,sample_type,idfold):

    accuracy = np.round(accuracy_score(y_test, prediction),4)
    recall = np.round(recall_score(y_test, prediction),4)
    precision = np.round(precision_score(y_test, prediction),4)
    f1score = np.round(f1_score(y_test, prediction), 4)
    gmean = np.round(geometric_mean_score(y_test, prediction, pos_label=1), 4)
    roc_auc = np.round(roc_auc_score(y_test, prediction),4)
    #kappa_metric = np.round(cohen_kappa_score(y_test, prediction),4)



    df = pd.DataFrame({
        "Dataset": dataset_name,
        "Encoder": encoder,
        "Validation_Type":validation_type,
        "Sample_Type":sample_type,
        "Fold": idfold,
        "Train_size": trainsize,
        "Train_shape": trainshape,
        "Accuracy": [accuracy],
        "Recall": [recall],
        "Precision": [precision],
        "f1-score": [f1score],
        "Roc_auc": [roc_auc],
        #"Kappa_metric": [kappa_metric],
        "G-mean": [gmean]

    })


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
        print('\n')

    return df



