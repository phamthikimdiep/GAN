# ----------------------------//-------------------------IMPORT LIBRARIES
# Basic import
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import pickle

import matplotlib.patches
from imblearn.metrics import geometric_mean_score
from numpy import ceil
import numpy as np
import pandas as pd
import itertools
import time
from scipy.stats import norm
import warnings
import math

warnings.filterwarnings("ignore")

# Plotting
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from IPython.display import SVG, display
from sklearn.tree import export_graphviz
import plotly.offline as py
import plotly.subplots as make_subplots
import plotly.figure_factory as ff  # visualization
import plotly.io as pio
import plotly.graph_objects as go
from yellowbrick.classifier import DiscriminationThreshold
from tabulate import tabulate
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, f_classif, SelectKBest, chi2

# Metrics
from sklearn import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import statsmodels.api as sm
from sklearn import tree

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from numpy import mean
from math import sqrt
from operator import itemgetter
import progressbar
import sys
from time import sleep
from impyute.imputation.cs import mice

from pandas import ExcelWriter

# Model Tuning
from bayes_opt import BayesianOptimization

# ----------------------------//-------------------------FUNCTIONS

# to view details of data
def detail_data(dataset, message):
    #print(f'{message}: \n')
    print('Rows: ', dataset.shape[0])
    print('\n Number of features: ', dataset.shape[1])
    print('\n Features:')
    print(dataset.columns.tolist())
    print('\n Missing values:', dataset.isnull().sum().values.sum())
    print('\n Unique values:')
    print(dataset.nunique())
    print('\n -------------Details of Dataset: {} -------------\n'.format(f'{message}'))
    t = 0
    dt_features = dataset.columns
    for i in dt_features:
        t = t + 1
        print('{} - {}'.format(t, i))
        uni = dataset[i].unique()
        print('unique values: {}'.format(uni))
        print('Counts: {}'.format(len(uni)))
        print("-" * 100)


# to display Confusion matrix Chart
#

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.xticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # else:
    # print('Confusion matrix without normalization')

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def plot_all_distribution_bar(df_value_counts, title, xlabel,ylabel):
    plt.figure(figsize=(8, 7))
    count = df_value_counts
    print("\n",count)
    sns.set(style="darkgrid")
    sns.barplot(count.index, count.values, alpha=0.7)
    for index, data in enumerate(count):
        plt.text(index, data + 3, s=f"{data}", fontdict=dict(fontsize=12))
    plt.title(title, fontsize=13)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.savefig('results_fig/' + title + '.png')
    #plt.show()


def plot_all_distribution_pie(title, sizes, colors, labels):
    plt.figure(figsize=(6,5))
    #explode = [0,0,0.1]
    plt.pie(sizes,colors=colors,labels=labels, shadow=True,autopct='%.2f%%', startangle = 0)
    plt.title(title + '\n',fontsize=13)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results_fig/'+ title + '.png')
    #plt.show()

def plot_all_distribution_hist(column_data, title, xlabel, ylabel,color):
    plt.figure(figsize=(6,5))
    plt.hist(column_data, color=color)
    plt.title(title, fontsize=13)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel,fontsize=12)
    plt.savefig('results_fig/' + title + '.png')
    #plt.show()

def plot_all_distribution_boxen(col_y, col_x, title, xlabel, ylabel, palette):
    plt.figure(figsize=(7,6))
    sns.boxenplot(col_y, col_x, palette=palette)
    plt.title(title, fontsize=13)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig('results_fig/' + title + '.png')
    #plt.show()

def plot_all_distribution_crosstab(col_x, col_y, title, figsize,colors):
    df = pd.crosstab(col_x, col_y)
    df.div(df.sum(1).astype(float),axis=0).plot(kind='bar',
                                                stacked = True,
                                                figsize=figsize,
                                                color=colors)

    plt.title(title,fontsize=13)
    plt.savefig('results_fig/'+title+'.png')
    #plt.show()

def save_model(model, out_dir):
    pickle.dump(model, open(out_dir, 'wb'))

def load_model(model_file):
    with open(model_file, 'rb') as file:
        return pickle.load(file)

def savefile_csv(target_file, saved_df):
    with open(target_file, 'w') as f_out:
        np.savetxt(f_out, saved_df, delimiter=',')


# remove special characters
# remove(filename, '\/:*?"<>|')
def remove(value, deletechars):
    for c in deletechars:
        value = value.replace(c, '')
    return value

#clean dataset
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def convert(data):
    number = preprocessing.LabelEncoder()
    data['Month'] = number.fit_transform(data['Month'])
    data['VisitorType'] = number.fit_transform(data['VisitorType'])
    data = data.fillna(-9999)
    return data


def run_progressbar(ranges):
    for i in range(ranges):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        i_per = round(100 * float(i) / float(ranges), 0)
        # print('i: {} - i_per: {}'.format(i,i_per))
        text = '[{}] {}% row: {}'.format('=' * int(i_per), i_per, i)
        sys.stdout.write(text)
        # sys.stdout.write("[%-20s] %d%% %row: d%" % ('=' * int(i_per), i_per,i))
        sys.stdout.flush()
        sleep(0.25)
    print(" - Done")


def model_score(algorithm, testing_y, predictions, probabilities):
    # roc_auc_score
    print('Algorithm: ', type(algorithm).__name__)
    print('Classification report: \n', classification_report(testing_y, predictions))
    print('Accuracy score: ', accuracy_score(testing_y, predictions))

    model_roc_auc = roc_auc_score(testing_y, predictions)
    print('Area under Curve: \n', model_roc_auc, '\n')

    fpr, tpr, threshold = roc_curve(testing_y, probabilities[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.title('Confusion Maxtrix - {}'.format(type(algorithm).__name__))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # confusion matrix

    conf_matrix = confusion_matrix(testing_y, predictions)
    print("Confusion Matrix: \n", conf_matrix)
    # plt.figure()
    # plot_confusion_matrix(conf_matrix, classes, title='Confusion Maxtrix - {}'.format(type(algorithm).__name__))
    # plt.savefig('results_fig/ConfusionMatrix_{}'.format(type(algorithm).__name__))


def churn_prediction(algorithm, training_x, testing_x, training_y, testing_y, threshold_plot):
    # model 1
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    model_score(algorithm, testing_y, predictions, probabilities)

    if threshold_plot:
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(training_x, training_y)
        visualizer.show()


def treeplot(classifier, cols, classnames):
    # plot decision tree

    graph = Source(tree.export_graphviz(classifier, out_file=None,
                                        rounded=True, proportion=False,
                                        feature_names=cols,
                                        precision=2,
                                        class_names=classnames,
                                        filled=True))
    # display(graph)


#########################################################
#        Model performance metrics                      #
#########################################################
# gives model report in dataframe
def model_report(model, training_x, testing_x, training_y, testing_y, name, kind, fold, datasetname):
    print('Training Time: ')
    training_time_start = time.perf_counter()
    #print('-----Starting time: {}'.format(training_time_start))
    model = model.fit(training_x, training_y)
    training_time_end = time.perf_counter()
    training_time = training_time_end - training_time_start


    #print('-----Ending time: {}'.format(training_time_end))
    #print('-'*20)
    print('-----> Training time: {} seconds ~ {} hours / {} minutes'.format(round(training_time, 4),
                                                                    round(training_time/3600,1),
                                                                    round(training_time/60,1)))

    if kind == 'SMOTE' or kind == 'REMOVE_OUTLIERS' or kind == 'Random_Undersampling':
        save_model(model, 'model_2after_preprocessing_data/{}.pkl'.format(datasetname + '_' + name + '_' + str(fold)))
    else:
        save_model(model, 'model_1before_preprocessing_data/{}.pkl'.format(datasetname + '_' + name + '_' + str(fold)))

    print('\nPrediction Time: ')
    prediction_time_start = time.perf_counter()
    #print('----- Starting time: {}'.format(prediction_time_start))
    predictions = model.predict(testing_x)
    prediction_time_end = time.perf_counter()
    prediction_time = prediction_time_end - prediction_time_start


    #print('----- Ending time: {}'.format(prediction_time_end))
    #print('-'*20)
    print('------> Prediction time: {} seconds ~ {} hours / {} minutes\n'.format(round(prediction_time, 4),
                                                                         round(prediction_time/3600,1),
                                                                         round(prediction_time/60,1)))


    accuracy = accuracy_score(testing_y, predictions)
    recall = recall_score(testing_y, predictions)
    precision = precision_score(testing_y, predictions)
    roc_auc = roc_auc_score(testing_y, predictions)
    f1score = f1_score(testing_y, predictions)
    kappa_metric = cohen_kappa_score(testing_y, predictions)
    gmean = geometric_mean_score(testing_y, predictions, pos_label = 1)
    df = pd.DataFrame({"Fold": [fold],
                       "Model": [name],
                       "Accuracy": [accuracy],
                       "Recall": [recall],
                       "Precision": [precision],
                       "f1-score": [f1score],
                       "Roc_auc": [roc_auc],
                       "Kappa_metric": [kappa_metric],
                       "G-mean": [gmean],
                       "Training time": [training_time],
                       "Prediction time": [prediction_time],
                       })
    with pd.option_context('display.max_rows',None,'display.max_columns',None):
        print(df)
        print('\n')
    return df


def euclidean_distance(a, b):
    return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(a, b)))


#########################################################
#        Compare model metrics                          #
#########################################################

def output_tracer(df, metric, color):
    tracer = go.Bar(y=df["Model"],
                    x=df[metric],
                    orientation="h", name=metric,
                    marker=dict(line=dict(width=.7), color=color)
                    )
    return tracer



def modelmetricsplot(df, title):
    layout = go.Layout(dict(title=title,
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            xaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       title="metric",
                                       zerolinewidth=1,
                                       ticklen=5, gridwidth=2),
                            yaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       zerolinewidth=1, ticklen=5, gridwidth=2),
                            margin=dict(l=250),
                            height=780
                            )
                       )
    trace1 = output_tracer(df, "Accuracy", "#6699FF")
    trace2 = output_tracer(df, 'Recall', "red")
    trace3 = output_tracer(df, 'Precision', "#33CC99")
    trace4 = output_tracer(df, 'f1-score', "lightgrey")
    trace5 = output_tracer(df, 'Roc_auc', "magenta")
    trace6 = output_tracer(df, 'Kappa_metric', "#FFCC99")

    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


########################################
# CONFUSION MATRIX                      #
########################################

def confmatplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber, kind, dataset_name, fold, labels):
    fig = plt.figure(figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber)))
    fig.set_facecolor("#F3F3F3")
    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        print('---', name)
        plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)

        if kind == 'SMOTE' or kind == 'Random_Undersampling' or kind == 'REMOVE_OUTLIERS':
            model = modeldict[name][0].fit(df_train[1][modeldict[name][1]], target_train[1])
            predictions = model.predict(df_test[modeldict[name][1]])
            conf_matrix = confusion_matrix(target_test, predictions)
            sns.heatmap(conf_matrix, annot=True, fmt="d", square=True,
                        xticklabels=labels,
                        yticklabels=labels,
                        linewidths=2, linecolor="w", cmap="Set1")
            plt.title(name, color="b")
            plt.subplots_adjust(wspace=.3, hspace=.3)

        else:
            model = modeldict[name][0].fit(df_train[0][modeldict[name][1]], target_train[0])
            predictions = model.predict(df_test[modeldict[name][1]])
            conf_matrix = confusion_matrix(target_test, predictions)
            sns.heatmap(conf_matrix, annot=True, fmt="d", square=True,
                        xticklabels=labels,
                        yticklabels=labels,
                        linewidths=2, linecolor="w", cmap="Set1")
            plt.title(name, color="b")
            plt.subplots_adjust(wspace=.3, hspace=.3)
        plt.savefig('results_fig/' + dataset_name + '_confusionmatrix_' + kind + '_' + str(fold) + '.png')


########################################
# ROC - Curves for models               #
########################################

def rocplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber, kind, dataset_name, fold):
    fig = plt.figure(figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber)))
    fig.set_facecolor("#F3F3F3")
    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        print('---', name)
        qx = plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)
        if kind == 'SMOTE' or kind == 'Random_Undersampling' or kind == 'REMOVE_OUTLIERS':
            model = modeldict[name][0].fit(df_train[1][modeldict[name][1]], target_train[1])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            fpr, tpr, thresholds = roc_curve(target_test, probabilities[:, 1])
            plt.plot(fpr, tpr, linestyle="dotted",
                     color="royalblue", linewidth=2,
                     label="AUC = " + str(np.around(roc_auc_score(target_test, predictions), 3)))
            plt.plot([0, 1], [0, 1], linestyle="dashed",
                     color="orangered", linewidth=1.5)
            plt.fill_between(fpr, tpr, alpha=.1)
            plt.fill_between([0, 1], [0, 1], color="b")
            plt.legend(loc="lower right",
                       prop={"size": 12})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xticks(np.arange(0, 1, .3))
            plt.yticks(np.arange(0, 1, .3))


        else:
            model = modeldict[name][0].fit(df_train[0][modeldict[name][1]], target_train[0])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            fpr, tpr, thresholds = roc_curve(target_test, probabilities[:, 1])
            plt.plot(fpr, tpr, linestyle="dotted",
                     color="royalblue", linewidth=2,
                     label="AUC = " + str(np.around(roc_auc_score(target_test, predictions), 3)))
            plt.plot([0, 1], [0, 1], linestyle="dashed",
                     color="orangered", linewidth=1.5)
            plt.fill_between(fpr, tpr, alpha=.1)
            plt.fill_between([0, 1], [0, 1], color="b")
            plt.legend(loc="lower right",
                       prop={"size": 12})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xticks(np.arange(0, 1, .3))
            plt.yticks(np.arange(0, 1, .3))
        plt.savefig('results_fig/' + dataset_name + '_roccurves_' + kind + '_' + str(fold) + '.png')


########################################
# Precision recall curves               #
########################################

def prcplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber, kind, dataset_name, fold):
    fig = plt.figure(figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber)))
    fig.set_facecolor("#F3F3F3")
    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        print('---', name)
        qx = plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)
        if kind == 'SMOTE' or kind == 'Random_Undersampling' or kind == 'REMOVE_OUTLIERS':
            model = modeldict[name][0].fit(df_train[1][modeldict[name][1]], target_train[1])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            recall, precision, thresholds = precision_recall_curve(target_test, probabilities[:, 1])
            plt.plot(recall, precision, linewidth=1.5,
                     label=("avg_pcn: " + str(np.around(average_precision_score(target_test, predictions), 3))))
            plt.plot([0, 1], [0, 0], linestyle="dashed")
            plt.fill_between(recall, precision, alpha=.1)
            plt.legend(loc="lower left", prop={"size": 10})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xlabel("recall", fontsize=7)
            plt.ylabel("precision", fontsize=7)
            plt.xlim([0.25, 1])
            plt.yticks(np.arange(0, 1, .3))

        else:
            model = modeldict[name][0].fit(df_train[0][modeldict[name][1]], target_train[0])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            recall, precision, thresholds = precision_recall_curve(target_test, probabilities[:, 1])
            plt.plot(recall, precision, linewidth=1.5,
                     label=("avg_pcn: " + str(np.around(average_precision_score(target_test, predictions), 3))))
            plt.plot([0, 1], [0, 0], linestyle="dashed")
            plt.fill_between(recall, precision, alpha=.1)
            plt.legend(loc="lower left", prop={"size": 10})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xlabel("recall", fontsize=7)
            plt.ylabel("precision", fontsize=7)
            plt.xlim([0.25, 1])
            plt.yticks(np.arange(0, 1, .3))
        plt.savefig('results_fig/' + dataset_name + '_precision_' + kind + '_' + str(fold) + '.png')


# -------------- Classifiers -----------------
# Baseline model
logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)

# LOGISTIC REGRESSION - SMOTE
logit_smote = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                 intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                 penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                 verbose=0, warm_start=False)

#remove outlier
logit_routliers = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                 intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                 penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                 verbose=0, warm_start=False)


# LOGISTIC REGRESSION - Random Undersampling
logit_rus = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)

# base
decision_tree = DecisionTreeClassifier(max_depth=9,
                                       random_state=123,
                                       splitter="best",
                                       criterion="gini")

# smote
decision_tree_smote = DecisionTreeClassifier(max_depth=9,
                                             random_state=123,
                                             splitter="best",
                                             criterion="gini")
#remove outlier
decision_tree_routliers  = DecisionTreeClassifier(max_depth=9,
                                                 random_state=123,
                                                 splitter="best",
                                                 criterion="gini")

# rus
decision_tree_rus = DecisionTreeClassifier(max_depth=9,
                                           random_state=123,
                                           splitter="best",
                                           criterion="gini")

# base
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                           weights='uniform')

# smote
knn_smote = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                                 weights='uniform')

#remove outlier
knn_routliers = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                    metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                                    weights='uniform')


# rus
knn_rus = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='uniform')

# base
rf = RandomForestClassifier(n_estimators=100, random_state=123,
                            max_depth=9, criterion="gini")

# smote
rf_smote = RandomForestClassifier(n_estimators=100, random_state=123,
                                  max_depth=9, criterion="gini")

#remove outlier
rf_routliers = RandomForestClassifier(n_estimators=100, random_state=123,
                                     max_depth=9, criterion="gini")


# rus
rf_rus = RandomForestClassifier(n_estimators=100, random_state=123,
                                max_depth=9, criterion="gini")

# base
nb = GaussianNB(priors=None)

# smote
nb_smote = GaussianNB(priors=None)

#remove outlier
nb_routliers = GaussianNB(priors=None)

# rus
nb_rus = GaussianNB(priors=None)

# LightGBM Classifier_base
lgbmc = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                       learning_rate=0.5, max_depth=7, min_child_samples=20,
                       min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                       n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                       reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                       subsample_for_bin=200000, subsample_freq=0)

# LightGBM Classifier_SMOTE
lgbmc_smote = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                             learning_rate=0.5, max_depth=7, min_child_samples=20,
                             min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                             n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                             reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                             subsample_for_bin=200000, subsample_freq=0)

#remove outlier
lgbmc_routliers = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                             learning_rate=0.5, max_depth=7, min_child_samples=20,
                             min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                             n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                             reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                             subsample_for_bin=200000, subsample_freq=0)

# LightGBM Classifier_rus
lgbmc_rus = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                           learning_rate=0.5, max_depth=7, min_child_samples=20,
                           min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                           n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                           reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                           subsample_for_bin=200000, subsample_freq=0)

# XGBoost Classifier_base
xgc = XGBClassifier(base_score=0.5, booster='gbtree',
                    colsample_bylevel=1, colsample_bytree=1,
                    gamma=0, learning_rate=0.9,
                    max_delta_step=0, max_depth=10,
                    min_child_weight=1, n_estimators=100,
                    n_jobs=1, nthread=None,
                    objective='binary:logistic',
                    random_state=0, reg_alpha=0,
                    reg_lambda=1, scale_pos_weight=1,
                    seed=None, subsample=1, eval_metric='logloss')

# XGBoost Classifier_smote
xgc_smote = XGBClassifier(base_score=0.5, booster='gbtree',
                          colsample_bylevel=1, colsample_bytree=1,
                          gamma=0, learning_rate=0.9,
                          max_delta_step=0, max_depth=10,
                          min_child_weight=1, n_estimators=100,
                          n_jobs=1, nthread=None,
                          objective='binary:logistic',
                          random_state=0, reg_alpha=0,
                          reg_lambda=1, scale_pos_weight=1,
                          seed=None, subsample=1, eval_metric='logloss')


#remove outlier
xgc_routliers = XGBClassifier(base_score=0.5, booster='gbtree',
                          colsample_bylevel=1, colsample_bytree=1,
                          gamma=0, learning_rate=0.9,
                          max_delta_step=0, max_depth=10,
                          min_child_weight=1, n_estimators=100,
                          n_jobs=1, nthread=None,
                          objective='binary:logistic',
                          random_state=0, reg_alpha=0,
                          reg_lambda=1, scale_pos_weight=1,
                          seed=None, subsample=1, eval_metric='logloss')


# XGBoost Classifier_rus
xgc_rus = XGBClassifier(base_score=0.5, booster='gbtree',
                        colsample_bylevel=1, colsample_bytree=1,
                        gamma=0, learning_rate=0.9,
                        max_delta_step=0, max_depth=10,
                        min_child_weight=1, n_estimators=100,
                        n_jobs=1, nthread=None,
                        objective='binary:logistic',
                        random_state=0, reg_alpha=0,
                        reg_lambda=1, scale_pos_weight=1,
                        seed=None, subsample=1, eval_metric='logloss')

# Gaussian Process Classifier
#kernel=1.0 * RBF(length_scale=1.0),optimizer=None
from sklearn.gaussian_process.kernels import RBF
gpc = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                random_state=0,
                                optimizer=None,
                                multi_class='one_vs_rest',
                                n_jobs = -1)

# Gaussian Process Classifier_smote
gpc_smote = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                      random_state=0,
                                      optimizer=None,
                                      multi_class='one_vs_rest',
                                      n_jobs = -1)

#remove outlier
gpc_routliers = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                          random_state=0,
                                          optimizer=None,
                                          multi_class='one_vs_rest',
                                          n_jobs = -1)

# Gaussian Process Classifier_rus
gpc_rus = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                    random_state=0,
                                    optimizer=None,
                                    multi_class='one_vs_rest',
                                    n_jobs = -1)

# AdaBoost Classifier_base
adac = AdaBoostClassifier(random_state=124)

# AdaBoost Classifier_smote
adac_smote = AdaBoostClassifier(random_state=124)

#remove outlier
adac_routliers = AdaBoostClassifier(random_state=124)

# AdaBoost Classifier_rus
adac_rus = AdaBoostClassifier(random_state=124)

# GradientBoosting Classifier_base
gbc = GradientBoostingClassifier(random_state=124)

# GradientBoosting Classifier_smote
gbc_smote = GradientBoostingClassifier(random_state=124)

#remove outlier
gbc_routliers = GradientBoostingClassifier(random_state=124)

# GradientBoosting Classifier_rus
gbc_rus = GradientBoostingClassifier(random_state=124)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()

# Linear Discriminant Analysis_smote
lda_smote = LinearDiscriminantAnalysis()

#remove outlier
lda_routliers = LinearDiscriminantAnalysis()

# Linear Discriminant Analysis_rus
lda_rus = LinearDiscriminantAnalysis()

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()

# Quadratic Discriminant Analysis_smote
qda_smote = QuadraticDiscriminantAnalysis()

#remove outlier
qda_routliers = QuadraticDiscriminantAnalysis()

# Quadratic Discriminant Analysis_rus
qda_rus = QuadraticDiscriminantAnalysis()

# Multi-layer Perceptron Classifier
mlp = MLPClassifier(alpha=1, max_iter=1000, random_state=124)

# Multi-layer Perceptron Classifier_smote
mlp_smote = MLPClassifier(alpha=1, max_iter=1000, random_state=124)

#remove outlier
mlp_routliers = MLPClassifier(alpha=1, max_iter=1000, random_state=124)

# Multi-layer Perceptron Classifier_rus
mlp_rus = MLPClassifier(alpha=1, max_iter=1000, random_state=124)

# Bagging Classifier
bgc = BaggingClassifier(random_state=124)

# Bagging Classifier_smote
bgc_smote = BaggingClassifier(random_state=124)

#remove outlier
bgc_routliers = BaggingClassifier(random_state=124)

# Bagging Classifier_rus
bgc_rus = BaggingClassifier(random_state=124)

# Support vector classifier using linear hyper plane
svc_lin = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
              max_iter=-1, probability=True, random_state=None, shrinking=True,
              tol=0.001, verbose=True)

# Support vector classifier using linear hyper plane_smote
svc_lin_smote = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
                    max_iter=-1, probability=True, random_state=None, shrinking=True,
                    tol=0.001, verbose=True)

#remove outlier
svc_lin_routliers = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                       decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
                       max_iter=-1, probability=True, random_state=None, shrinking=True,
                       tol=0.001, verbose=True)

# Support vector classifier using linear hyper plane_rus
svc_lin_rus = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
                  max_iter=-1, probability=True, random_state=None, shrinking=True,
                  tol=0.001, verbose=True)

# support vector classifier using non-linear hyper plane ("rbf")
svc_rbf = SVC(C=1.0, kernel='rbf',
              degree= 3, gamma=1.0,
              coef0=0.0, shrinking=True,
              probability=True, tol=0.001,
              cache_size=200, class_weight=None,
              verbose=True,max_iter= -1,
              random_state=None)

# support vector classifier using non-linear hyper plane ("rbf")_smote
svc_rbf_smote = SVC(C=1.0, kernel='rbf',
                    degree= 3, gamma=1.0,
                    coef0=0.0, shrinking=True,
                    probability=True, tol=0.001,
                    cache_size=200, class_weight=None,
                    verbose=True,max_iter= -1,
                    random_state=None)

#remove outlier
svc_rbf_routliers= SVC(C=1.0, kernel='rbf',
                      degree= 3, gamma=1.0,
                      coef0=0.0, shrinking=True,
                      probability=True, tol=0.001,
                      cache_size=200, class_weight=None,
                      verbose=True,max_iter= -1,
                      random_state=None)


# support vector classifier using non-linear hyper plane ("rbf")_rus
svc_rbf_rus = SVC(C=1.0, kernel='rbf',
                  degree= 3, gamma=1.0,
                  coef0=0.0, shrinking=True,
                  probability=True, tol=0.001,
                  cache_size=200, class_weight=None,
                  verbose=True,max_iter= -1,
                  random_state=None)
