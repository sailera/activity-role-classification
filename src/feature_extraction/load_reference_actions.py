import os
import pandas as pd
from src.constants import FILEPATH_OTHER


def load_reference_actions(include_levin=False) -> list[tuple[str, list[str]]]:
    """
    Load the reference actions from the MIT (and Levin) scheme to the combined new scheme.

    :param include_levin: whether to include the verbs from levin verb index as references
    :return: list of (action, [reference_actions])
    """
    # load from MIT scheme
    df_reference = pd.read_csv(os.path.join(FILEPATH_OTHER, 'mit_referenceactions.csv'), dtype=str)
    df_reference['reference:actions'] = df_reference['reference:actions'].apply(lambda x: x.split(', '))

    # combine to new scheme
    reference_actions = {}
    for _, _, ref, category in df_reference.itertuples():
        if category not in reference_actions.keys():
            reference_actions[category] = []
        reference_actions[category].extend(ref)

    if include_levin:
        # load from Levin scheme
        df_index = pd.read_csv(os.path.join(FILEPATH_OTHER, 'levin_verbindex.csv'))
        df_index['occurrences'] = df_index['occurrences'].apply(lambda x: x.split(', '))
        df_classes = pd.read_csv(os.path.join(FILEPATH_OTHER, 'levin_verbclasses.csv'), dtype=str)

        for _, index, _, category in df_classes.itertuples():
            if category not in reference_actions.keys():
                reference_actions[category] = []
            for _, verb, occurrences in df_index.itertuples():
                for occurrence in occurrences:
                    if occurrence.startswith(index) or occurrence == index.strip('.'):
                        reference_actions[category].append(verb)

    reference_actions = list(reference_actions.items())
    return reference_actions


if __name__ == '__main__':
    ref_actions = load_reference_actions(include_levin=True)
    for i in ref_actions:
        if i[0] == 'communicate':
            print(i[1])\
