from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from src.constants import RANDOM_SEED
from src.classification.utils import load_data, save_figures, predict_classifier, OptionalOversampler, \
    find_best_params, FeatureFilter


def evaluate_rfc():
    df, X, y, splits = load_data()
    pipe = Pipeline([
        ('filter', FeatureFilter()),
        ('selector', SelectKBest(mutual_info_classif)),
        ('ros', OptionalOversampler(random_state=RANDOM_SEED)),
        ('rfc', RandomForestClassifier(random_state=RANDOM_SEED))
    ])
    params = [
        {
            'filter__embeddings': [True],
            'selector__k': [i for i in range(20, 196)],
            'ros__activate': [True, False],
            'rfc__n_estimators': [200, 500, 1000],
            'rfc__max_features': ['sqrt', 'log2'],
            'rfc__max_depth': [5, 10, 20],
            'rfc__criterion': ['gini', 'entropy']
        },
        {
            'filter__embeddings': [False],
            'selector__k': [i for i in range(20, 96)],
            'ros__activate': [True, False],
            'rfc__n_estimators': [200, 500, 1000],
            'rfc__max_features': ['sqrt', 'log2'],
            'rfc__max_depth': [5, 10, 20],
            'rfc__criterion': ['gini', 'entropy']
        }
    ]
    score, best_params = find_best_params(pipe, X, y, splits, params, 'rfc')
    print(best_params)

    pipe.set_params(**best_params)
    y_pred = predict_classifier(pipe, X, y, df, splits, 'rfc')

    save_figures(y, y_pred, 'rfc')
    print(f1_score(y, y_pred, average='macro'))


if __name__ == '__main__':
    evaluate_rfc()
