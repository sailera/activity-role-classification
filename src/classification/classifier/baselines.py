from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from src.classification.utils import load_data, save_figures, predict_classifier


def evaluate_majority_baseline():
    df, X, y, splits = load_data()

    clf = DummyClassifier()
    y_pred = predict_classifier(clf, X, y, df, splits, 'majority_baseline')

    save_figures(y, y_pred, 'majority_baseline')
    print(f1_score(y, y_pred, average='macro'))


if __name__ == '__main__':
    evaluate_majority_baseline()
