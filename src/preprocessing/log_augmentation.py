import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer
from pm4py.objects.conversion.log import converter
from pm4py.objects.log.util import dataframe_utils
from extraction import extract

from src.constants import KNOWN_SEPARATORS, KNOWN_LOG_KEYS, KNOWN_ACTIVITY_KEYS, KNOWN_EVENT_TIME_KEYS, DEFAULT_CASE_KEY, \
    DEFAULT_ACTIVITY_KEY, DEFAULT_EVENT_TIME_KEY, ALL_EVENT_LOGS, FILEPATH_LOGS, FILEPATH_AUGMENTED, SEMANTIC_ROLES


def _load_log(filepath: str, filename: str) -> pd.DataFrame:
    """
    Loads an event log file (.csv or .xes) to a Dataframe.

    :param filepath: Filepath where the log file is stored
    :param filename: Filename of event log
    :return: Dataframe of event log
    """
    print('Load Event Log')
    event_log = None
    filename_only = filename[:-4]
    if filename.endswith('.xes'):
        event_log = importer.apply(os.path.join(filepath, filename))
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(filepath, filename), sep=KNOWN_SEPARATORS[filename_only])
        except UnicodeDecodeError:
            df = pd.read_csv(os.path.join(filepath, filename), sep=KNOWN_SEPARATORS[filename_only],
                             encoding="ISO-8859-1")
        try:
            df = dataframe_utils.convert_timestamp_columns_in_df(df)
        except TypeError:
            pass
        print('Convert Event Log')
        if filename_only in KNOWN_LOG_KEYS:
            event_log = converter.apply(df,
                                        parameters={
                                            converter.to_event_log.Parameters.CASE_ID_KEY: KNOWN_LOG_KEYS[filename_only]
                                        },
                                        variant=converter.Variants.TO_EVENT_LOG)
        else:
            event_log = converter.apply(df, variant=converter.Variants.TO_EVENT_LOG)

    # get keys for this       
    case_key = DEFAULT_CASE_KEY
    activity_key = KNOWN_ACTIVITY_KEYS.get(filename_only, DEFAULT_ACTIVITY_KEY)
    event_time_key = KNOWN_EVENT_TIME_KEYS.get(filename_only, DEFAULT_EVENT_TIME_KEY)

    return event_log, case_key, activity_key, event_time_key


def augment_log(filepath_from: str, filepath_to: str, filename: str) -> pd.DataFrame:
    """
    Augments an event log with semantic roles.

    :param filepath_from: Filepath where the log file is stored
    :param filepath_to: Filepath where the augmented log file should be stored
    :param filename: Filename of event log
    :return: Dataframe of augmented event log
    """
    event_log, case, activity, event_time = _load_log(filepath_from, filename)
    role_extractor = extract.get_instance()
    print("Start Augmentation")
    df_augmented = role_extractor.augment_event_log_with_semantic_roles(event_log, exp=False, as_df=True, conf=0.9,
                                                                        bo_ratio=0.5,
                                                                        case=case,
                                                                        activity=activity,
                                                                        event_time=event_time,
                                                                        name='log')

    df_augmented = df_augmented.fillna('')

    # define the case, activity and time columns from the original log
    stripped_filename = filename[:-4]
    activity_key = KNOWN_ACTIVITY_KEYS.get(stripped_filename, DEFAULT_ACTIVITY_KEY)
    event_time_key = KNOWN_EVENT_TIME_KEYS.get(stripped_filename, DEFAULT_EVENT_TIME_KEY)

    # set the id, activity and event_time from the original log
    df_augmented.rename(columns={activity_key: DEFAULT_ACTIVITY_KEY, event_time_key: DEFAULT_EVENT_TIME_KEY},
                        inplace=True)
    missing_semantic_roles = [role for role in SEMANTIC_ROLES if role not in df_augmented.columns]
    if len(missing_semantic_roles) > 0:
        df_augmented[missing_semantic_roles] = None
    df_augmented = df_augmented[[DEFAULT_CASE_KEY, DEFAULT_ACTIVITY_KEY, DEFAULT_EVENT_TIME_KEY, *SEMANTIC_ROLES]]

    # save df to file and return df
    df_augmented.to_csv(os.path.join(filepath_to, stripped_filename + '.csv'), index=False)
    return df_augmented


def augment_all_logs() -> None:
    """
    Augment all event logs in the given input directory.

    :return: None
    """
    print('Started Event Log Augmentation.\n')
    for log_filename in ALL_EVENT_LOGS:
        print('\nEvent Log Augmentation for "{0}" ...'.format(log_filename))
        augment_log(FILEPATH_LOGS, FILEPATH_AUGMENTED, log_filename)
    print('\n\nDone.')


if __name__ == '__main__':
    augment_all_logs()
