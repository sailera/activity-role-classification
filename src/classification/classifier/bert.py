import torch
import warnings
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline
from transformers import logging as hf_logging
from simpletransformers.classification import ClassificationModel

from src.constants import DEFAULT_ACTIVITY_KEY, FILEPATH_MODELS_BERT, RANDOM_SEED
from src.classification.utils import load_data, find_best_params, OptionalOversampler, save_figures, predict_classifier

cuda_available = torch.cuda.is_available()


class BertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, train_batch_size=None, num_train_epochs=None, learning_rate=None, max_seq_length=16,
                 random_state=RANDOM_SEED):
        self.train_batch_size = train_batch_size
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.random_state = random_state
        self.y_ = None
        self.X_ = None
        self.classes_ = None
        self.class_dict_ = {}
        self.model = None

    def fit(self, X, y):
        # disable warnings
        warnings.filterwarnings('ignore')
        hf_logging.set_verbosity_error()

        # store the classes seen during fit
        possible_labels = y.unique()

        self.class_dict_ = {}
        for index, possible_label in enumerate(possible_labels):
            self.class_dict_[possible_label] = index

        y = y.replace(self.class_dict_)

        self.classes_ = list(self.class_dict_.keys())
        self.X_ = X
        self.y_ = y
        train_df = pd.DataFrame(X[DEFAULT_ACTIVITY_KEY])
        train_df['label'] = y

        args = {
            'output_dir': FILEPATH_MODELS_BERT,
            'overwrite_output_dir': True,
            'silent': True,
            'random_state': self.random_state,
            'max_seq_length': self.max_seq_length
        }
        if self.train_batch_size:
            args['train_batch_size'] = self.train_batch_size
        if self.num_train_epochs:
            args['num_train_epochs'] = self.num_train_epochs
        if self.learning_rate:
            args['learning_rate'] = self.learning_rate

        self.model = ClassificationModel('bert', 'bert-base-uncased', num_labels=len(self.classes_), args=args,
                                         use_cuda=cuda_available)
        self.model.train_model(train_df)
        # Return the classifier
        return self

    def predict(self, X):
        test = list(X[DEFAULT_ACTIVITY_KEY])
        pred, _ = self.model.predict(test)
        class_dict_reverse = {v: k for k, v in self.class_dict_.items()}
        return [class_dict_reverse[p] for p in pred]


def evaluate_bert():
    X, _, y, splits = load_data()

    pipe = Pipeline([
        ('ros', OptionalOversampler(random_state=RANDOM_SEED)),
        ('bert', BertClassifier(random_state=RANDOM_SEED))
    ])
    params = {
        'ros__activate': [True, False],
        'bert__train_batch_size': [16],#[16, 32],
        'bert__num_train_epochs': [2],#[2, 3, 4],
        'bert__learning_rate': [5e-5],#[5e-5, 3e-5, 2e-5],
    }

    score, best_params = find_best_params(pipe, X, y, splits, params, 'bert', n_jobs=1)
    print(best_params)

    pipe.set_params(**best_params)
    y_pred = predict_classifier(pipe, X, y, X, splits, 'bert')

    save_figures(y, y_pred, 'bert')
    print(f1_score(y, y_pred, average='macro'))


if __name__ == '__main__':
    evaluate_bert()
