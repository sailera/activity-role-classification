import os
import pandas as pd

from src.constants import ANNOTATION_KEY, ANNOTATION_LABELS, DEFAULT_ACTIVITY_KEY, DEFAULT_CASE_KEY, \
    EVENT_LOG_NAME_KEY, ALL_EVENT_LOG_NAMES, FILEPATH_PREPROCESSED, FILE_ACTIVITY_ROLE_ANNOTATIONS


def create_annotation_file() -> pd.DataFrame:
    """Create a file to annotate the samples of all given event logs.

    :return: Dataframe of the samples to annotate
    """
    df_annotate = None
    for i, filename in enumerate(ALL_EVENT_LOG_NAMES):
        # load file to df
        df = pd.read_csv(os.path.join(FILEPATH_PREPROCESSED, filename + '.csv'), low_memory=False)

        # deduplicate based on event class (unique activity labels, keep first sample of this event class)
        df = df.drop_duplicates(subset=[DEFAULT_ACTIVITY_KEY]).sort_values(by=DEFAULT_ACTIVITY_KEY)

        # add a column for the event log name and for the manual annotation
        df.insert(2, EVENT_LOG_NAME_KEY, filename)
        df.insert(0, ANNOTATION_KEY, '')

        # concat the event logs
        if i == 0:
            df_annotate = df
        else:
            df_annotate = pd.concat([df_annotate, df])

    # remove the case column
    df_annotate = df_annotate.drop(columns=[DEFAULT_CASE_KEY])

    # save df to file and return df
    df_annotate.to_csv(os.path.join(FILE_ACTIVITY_ROLE_ANNOTATIONS), index=False)
    return df_annotate


def get_gold_standard() -> pd.DataFrame:
    """Load the annotated samples to a DataFrame.

    :return: Dataframe of the annotated samples
    """
    # load the manual annotations
    df_annotations = pd.read_csv(os.path.join(FILE_ACTIVITY_ROLE_ANNOTATIONS), low_memory=False)

    # check for wrong annotations (e.g. "comunicate" instead of "communicate")
    for item in df_annotations[ANNOTATION_KEY]:
        if item not in ANNOTATION_LABELS:
            raise ValueError('{0} is not a valid annotation'.format(item))

    # return the annotations as gold standard
    return df_annotations


if __name__ == '__main__':
    if not os.path.exists(FILE_ACTIVITY_ROLE_ANNOTATIONS):
        create_annotation_file()
    else:
        df_gold = get_gold_standard()
        print(df_gold.head())
