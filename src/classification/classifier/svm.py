from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from src.constants import RANDOM_SEED
from src.classification.utils import load_data, save_figures, predict_classifier, OptionalOversampler,\
    FeatureFilter, find_best_params


def evaluate_svm():
    df, X, y, splits = load_data()
    pipe = Pipeline([
        ('filter', FeatureFilter()),
        ('selector', SelectKBest(mutual_info_classif)),
        ('scaler', StandardScaler()),
        ('ros', OptionalOversampler(random_state=RANDOM_SEED)),
        ('svc', SVC(random_state=RANDOM_SEED))
    ])
    params = [
        {
            'filter__embeddings': [True],
            'selector__k': [i for i in range(20, 196)],
            'ros__activate': [True, False],
            'svc__C': [1, 10, 100, 1000],
            'svc__gamma': [0.0001, 0.001, 0.01, 0.1],
            'svc__kernel': ['rbf']
        },
        {
            'filter__embeddings': [False],
            'selector__k': [i for i in range(20, 96)],
            'ros__activate': [True, False],
            'svc__C': [1, 10, 100, 1000],
            'svc__gamma': [0.0001, 0.001, 0.01, 0.1],
            'svc__kernel': ['rbf']
        }
    ]

    score, best_params = find_best_params(pipe, X, y, splits, params, 'svc')
    print(best_params)

    pipe.set_params(**best_params)
    y_pred = predict_classifier(pipe, X, y, df, splits, 'svc')

    save_figures(y, y_pred, 'svc')
    print(f1_score(y, y_pred, average='macro'))


if __name__ == '__main__':
    evaluate_svm()
