import pandas as pd
import numpy as np
import os
import random
from src.constants import FILEPATH_OTHER, FILEPATH_ANNOTATIONS, FILEPATH_USECASE

random.seed(42)


def create_annotation_file(filename):
    """Create a file for best-worst-scaling annotation

    :param str filename: Filename of the event log to annotate
    :return: Dataframe of the event log to annotate
    """
    # load file to df
    df = pd.read_csv(os.path.join(FILEPATH_OTHER, filename + '.csv'), low_memory=False, delimiter=';')
    items = []
    for idx, row in df.iterrows():
        if 'XOR' in row['anomaly type']:
            if '{0} occurred together with {1}'.format(row['event2'], row['event1']) in items:
                continue
            items.append('{0} occurred together with {1}'.format(row['event1'], row['event2']))
        elif 'CO_OCC' in row['anomaly type']:
            if '{0} occurred without {1}'.format(row['event2'], row['event1']) in items:
                continue
            items.append('{0} occurred without {1}'.format(row['event1'], row['event2']))
        elif 'ORDER' in row['anomaly type']:
            items.append('{0} occurred before {1}'.format(row['event1'], row['event2']))
    tuples = []
    items_remaining = items
    for i in range(len(items) * 2):
        if len(items_remaining) < 4:
            items_remaining = items
        random.shuffle(items_remaining)
        tuples.append(items_remaining[0:4])
        items_remaining = items_remaining[4:]
    df_tuples = pd.DataFrame(tuples, columns=['item:{0}'.format(i+1) for i in range(4)])
    df_tuples.insert(0, 'least', '', True)
    df_tuples.insert(0, 'most', '', True)
    df_tuples.to_csv(os.path.join(FILEPATH_ANNOTATIONS, filename + '_annotations.csv'), index=False)


def compute_scores(filename):
    df = pd.read_csv(os.path.join(FILEPATH_ANNOTATIONS, filename + '_annotations.csv'), low_memory=False)
    scores = {}
    for _, row in df.iterrows():
        least_severe = row['least']
        most_severe = row['most']
        for i in range(1, 5):
            item = row['item:{0}'.format(i)]
            if item not in scores:
                scores[item] = []
            if i == least_severe:
                scores[item].append(-1)
            elif i == most_severe:
                scores[item].append(1)
            else:
                scores[item].append(0)
    scores = [(k, np.array(v).mean()) for k, v in scores.items()]
    df_scores = pd.DataFrame(scores, columns=['violation', 'severity'])
    df_scores = df_scores.sort_values('severity')

    def _get_events(row):
        if ' occurred together with ' in row:
            violation = 'XOR'
            event1, event2 = row.split(' occurred together with ')
        elif ' occurred without ' in row:
            violation = 'CO_OCC'
            event1, event2 = row.split(' occurred without ')
        else:
            violation = 'ORDER'
            event1, event2 = row.split(' occurred before ')
        return pd.Series([violation, event1, event2])

    df_scores[['type', 'event1', 'event2']] = df_scores['violation'].apply(lambda x: _get_events(x))
    df_scores.to_csv(os.path.join(FILEPATH_USECASE, filename + '_scores.csv'), index=False)


if __name__ == '__main__':
    # create_annotation_file('results_anomaly_bpi_2018')
    compute_scores('results_anomaly_bpi_2018')
