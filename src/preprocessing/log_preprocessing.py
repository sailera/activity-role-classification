import os
import nltk as nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm

from src.constants import ALL_EVENT_LOG_NAMES, FILEPATH_AUGMENTED, FILEPATH_PREPROCESSED

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def lemmatize(phrases: str, pos: str) -> str:
    """
    Lemmatize comma-separated words depending on given POS-Tag.

    :param phrases: comma-separated words to lemmatize
    :param pos: POS-Tag of the words (assumes same tag for all roles, e.g. verb for 'action:name' role)
    :return: str of comma-separated lemmata
    """
    if not isinstance(phrases, str):
        return phrases
    return ','.join([' '.join([lemmatizer.lemmatize(word, pos=pos) for word in phrase.split(' ')])
                     for phrase in phrases.split(',')])


def lemmatize_log(filepath_from: str, filepath_to: str, filename: str) -> pd.DataFrame:
    """
    Lemmatizes the semantic roles 'action:name' and 'object:name' of an event log.

    :param filepath_from: Filepath where the log file is stored
    :param filepath_to: Filepath where the lemmatized log file should be stored
    :param filename: Filename of event log
    :return: Dataframe of lemmatized event log
    """
    df_log = pd.read_csv(os.path.join(filepath_from, filename + '.csv'), low_memory=False)
    for role, pos in [('action:name', wordnet.VERB), ('object:name', wordnet.NOUN)]:
        tqdm.pandas(desc='Preprocessing {0}'.format(role))
        df_log[role] = df_log[role].progress_apply(lambda x: lemmatize(x, pos))
    df_log.to_csv(os.path.join(filepath_to, filename + '.csv'), index=False)
    return df_log


def lemmatize_all_logs() -> None:
    """
    Lemmatize all event logs in the given input directory.

    :return: None
    """
    for log_filename in ALL_EVENT_LOG_NAMES:
        print('Preprocessing for "{0}" ...'.format(log_filename))
        lemmatize_log(FILEPATH_AUGMENTED, FILEPATH_PREPROCESSED, log_filename)
        print('\n')


if __name__ == '__main__':
    lemmatize_all_logs()
