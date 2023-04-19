import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score

from src.constants import ANNOTATION_LABELS
from src.classification.utils import load_data, save_figures, predict_classifier


class ActionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.y_ = None
        self.X_ = None
        self.classes_ = None
        self.class_frequencies_ = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Store the classes seen during fit
        self.classes_ = ANNOTATION_LABELS
        for label in unique_labels(y):
            # y can be the gold standard labels or '' in case no gold standard is available
            if label not in self.classes_ and label != '':
                raise ValueError('The label {0} is not a valid category.'.format(label))

        # Get the class frequencies
        self.class_frequencies_ = pd.Series(0, index=ANNOTATION_LABELS)  # default frequency 0
        frequencies = pd.Series(y).value_counts()
        if '' in frequencies.index:
            frequencies = frequencies.drop(index=[''])  # remove '' (dummy label if no gold standard is given)
        self.class_frequencies_[frequencies.index] = frequencies  # overwrite with actual values

        # Return the classifier
        return self

    def predict(self, X):
        ref_cols = [c for c in X.columns if c.startswith('feature:similarity:action:name:max:ref')]

        def _get_pred(x):
            # get reference similarities
            x_ref = x[ref_cols]

            # get the max similarities
            max_val = x_ref.max()
            max_refs = x_ref[x_ref == max_val]
            role_cols = [c.replace('max:ref:', 'max:role:') for c in list(max_refs.index)]

            # there might be multiple with the same reference similarity, find the one with max role similarity
            x_role = x[role_cols]
            max_val = x_role.max()
            max_roles = x_role[x_role == max_val]
            max_roles = [role.split(':')[-1] for role in list(max_roles.index)]

            # if there is only one most similar class, predict it
            if len(max_roles) == 1:
                return max_roles[0]
            # otherwise, predict the one that is most frequent
            return self.class_frequencies_[max_roles].idxmax()

        return X.apply(lambda x: _get_pred(x), axis=1)


def evaluate_action_classifier():
    df, X, y, splits = load_data()

    clf = ActionClassifier()
    y_pred = predict_classifier(clf, X, y, df, splits, 'action_classifier')

    save_figures(y, y_pred, 'action_classifier')
    print(f1_score(y, y_pred, average='macro'))


if __name__ == '__main__':
    evaluate_action_classifier()
