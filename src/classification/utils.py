import os
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict, ParameterGrid
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import RandomOverSampler

from src.feature_extraction.load_reference_actions import load_reference_actions
from src.constants import FILEPATH_FEATURES, FILEPATH_PREDICTIONS, FILEPATH_GRIDSEARCH, FILEPATH_FIGURES, \
    DEFAULT_ACTIVITY_KEY, ANNOTATION_KEY, EVENT_LOG_NAME_KEY, FILE_ACTIVITY_ROLE_ANNOTATIONS


RANDOM_SEED = 42


f1_macro_score = make_scorer(f1_score, average='macro')


def load_data():
    """
    Get the dataset for training / evaluation.

    :return: df, X, y, splits
    """
    df_X = pd.read_csv(os.path.join(FILEPATH_FEATURES, 'all.csv'), low_memory=False)
    df_y = pd.read_csv(FILE_ACTIVITY_ROLE_ANNOTATIONS, low_memory=False)
    X = df_X.drop(columns=[EVENT_LOG_NAME_KEY, DEFAULT_ACTIVITY_KEY]).fillna(0)
    y = df_y[ANNOTATION_KEY]
    splits = get_event_log_splits(df_X)
    return df_X, X, y, splits


def get_event_log_splits(df):
    """
    Create splits for CV by event log name.

    :param df: pd.DataFrame, requires column 'event:log:name'
    :return: list of list of indices
    """
    splits = []
    for log in df[EVENT_LOG_NAME_KEY].unique():
        train = df[df[EVENT_LOG_NAME_KEY] != log].index
        test = df[df[EVENT_LOG_NAME_KEY] == log].index
        splits.append((train, test))
    return splits


def predict_classifier(clf, X, y, df, splits, filename):
    """

    :param clf: Classifier (sklearn or custom)
    :param X: pd.DataFrame
    :param y: pd.Series
    :param df: pd.DataFrame
    :param splits: list of list of indices
    :param filename: str
    :return: prediction (pd.Series)
    """
    filepath = os.path.join(FILEPATH_PREDICTIONS, filename + '.csv')
    if os.path.exists(filepath):
        df_pred = pd.read_csv(filepath)
    else:
        np.random.seed(RANDOM_SEED)
        y_pred = cross_val_predict(clf, X, y, cv=splits)
        df_pred = pd.DataFrame(y_pred, columns=['pred'])
        df_pred.insert(0, ANNOTATION_KEY, y)
        df_pred.insert(0, DEFAULT_ACTIVITY_KEY, df[DEFAULT_ACTIVITY_KEY])
        df_pred.to_csv(filepath, index=False)
    return df_pred['pred']


def find_best_params(pipe, X, y, splits, params, filename, random_state=RANDOM_SEED, n_jobs=-1):
    """
    Find the best parameter set (f1_score, average=macro) for a pipeline.

    :param pipe: Pipeline (imblearn or sklearn)
    :param X: pd.DataFrame
    :param y: pd.Series
    :param splits: list of list of indices
    :param params: dict
    :param filename: str
    :param random_state: int
    :param n_jobs: int
    :return: best f1-score, best params
    """
    filepath = os.path.join(FILEPATH_GRIDSEARCH, filename + '.pkl')
    if os.path.exists(filepath):
        results = pickle.load(open(filepath, 'rb'))
    else:
        pg = ParameterGrid(params)

        def _cross_val_predict(param_set):
            np.random.seed(random_state)
            clf = pipe.set_params(**param_set)
            y_pred = cross_val_predict(clf, X, y, cv=splits)
            score = f1_score(y, y_pred, average='macro')
            return score, param_set

        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_cross_val_predict)(param_set) for param_set in pg)
        pickle.dump(results, open(filepath, 'wb'))
    best_score, best_params = max(results, key=lambda item: item[0])
    return best_score, best_params


def save_figures(y, y_pred, filename):
    """
    Save a figure of the classification report and the confusion matrix.

    :param y: list
    :param y_pred: list
    :param filename: str
    """
    # create report plot
    report = classification_report(y, y_pred, output_dict=True)
    plt.figure()
    fig_report = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='viridis').get_figure()
    plt.tight_layout()
    fig_report.savefig(os.path.join(FILEPATH_FIGURES, './classification_report/' + filename + '.png'))

    # change labels to ints
    keys = [ref for ref, _ in load_reference_actions()]
    y = [keys.index(v) for v in y]
    y_pred = [keys.index(v) for v in y_pred]
    # create confusion matrix plot
    cm = confusion_matrix(y, y_pred)
    plt.figure()
    ax = plt.subplot()
    fig_cm = sns.heatmap(pd.DataFrame(cm), annot=True, ax=ax, fmt='g', cmap='viridis').get_figure()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    ax.set_xticklabels(keys, rotation=90)
    ax.set_yticklabels(keys, rotation=0)
    plt.tight_layout()
    fig_cm.savefig(os.path.join(FILEPATH_FIGURES, './confusion_matrix/' + filename + '.png'))


class OptionalOversampler(RandomOverSampler):
    def __init__(self, activate=True, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate

    def fit_resample(self, X, y):
        if self.activate:
            return super().fit_resample(X, y)
        else:
            return X, y


class FeatureFilter(TransformerMixin, BaseEstimator):
    def __init__(self, embeddings=True, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = embeddings

    def transform(self, X):
        if self.embeddings:
            return X
        else:
            other_cols = [col for col in X.columns if not col.startswith('feature:embedding:')]
            return X[other_cols]

    def fit(self, X, y=None):
        return self
