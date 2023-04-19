from tqdm import tqdm
import os
import re
import pandas as pd
import numpy as np
from src.classification.classifier.action_classifier import ActionClassifier
from src.feature_extraction.embeddings_loader import EmbeddingsLoader, cosine_similarity
from src.constants import ALL_EVENT_LOG_NAMES, DEFAULT_ACTIVITY_KEY, DEFAULT_CASE_KEY, DEFAULT_EVENT_TIME_KEY, \
    EVENT_LOG_NAME_KEY, FILEPATH_PREPROCESSED, FILEPATH_FEATURES, ANNOTATION_LABELS, ANNOTATION_KEY, SEMANTIC_ROLES


class Extractor:
    def __init__(self, filename, embeddings):
        self.embeddings = embeddings
        self.filename = filename

        # load preprocessed event log
        self.df = pd.read_csv(os.path.join(FILEPATH_PREPROCESSED, filename + '.csv'), low_memory=False)
        self.df = self.df.sort_values([DEFAULT_CASE_KEY, DEFAULT_EVENT_TIME_KEY])
        self.df = self.df.fillna('').reset_index()

        # initialize feature df
        self.df_features = self.df[[DEFAULT_CASE_KEY, DEFAULT_ACTIVITY_KEY]].copy()
        self.df_features.insert(0, EVENT_LOG_NAME_KEY, filename, True)

    def _extract_similarities(self) -> None:
        """
        Extract the cosine similarity of the event actions with each category.

        :return: None
        """
        def _get_max_similarity(words, reference_vectors_for_role):
            similarities = []
            for word in words:
                vec = self.embeddings.get(word)
                if vec is None:
                    continue
                for ref_vec in reference_vectors_for_role:
                    similarities.append(cosine_similarity(vec, ref_vec))
            if len(similarities) == 0:
                return 0
            return np.max(similarities)

        lookup = {}

        def _get_similarities(phrases, feature_prefix):
            phrases = re.sub('[^a-zA-Z, ]', '', phrases)
            cached = lookup.get(phrases, None)
            if cached is not None:
                return cached

            words = [word for phrase in phrases.split(',') for word in phrase.split(' ')]

            similarities = pd.Series(dtype='float64')
            for ref_role, ref_vectors_for_role in self.embeddings.reference_vectors:
                # get similarity only for main role (e.g. "transform")
                similarities[feature_prefix + 'max:role:' + ref_role] = _get_max_similarity(words,
                                                                                            ref_vectors_for_role[0:1])

                # get similarity for all reference actions of role
                similarities[feature_prefix + 'max:ref:' + ref_role] = _get_max_similarity(words,
                                                                                           ref_vectors_for_role)

            lookup[phrases] = similarities
            return similarities

        for sem_role in ['action:name', 'object:name']:
            lookup = {}  # reset lookup
            tqdm.pandas(desc='Extract Similarities {0}'.format(sem_role))
            prefix = 'feature:similarity:{0}:'.format(sem_role)
            df_similarities = self.df[sem_role].progress_apply(lambda c: _get_similarities(c, prefix))
            self.df_features = pd.concat([self.df_features, df_similarities], axis=1)

    def _extract_action_category(self) -> None:
        """
        Perform the "simple" action categorization (based on reference actions) and assign it to the events.

        :return: None
        """
        print('Extracting action category ...')
        # transform the data to only contain the similarity features (each unique combination once)
        similarity_features = [column for column in self.df_features.columns if column.startswith('feature:similarity')]
        X = self.df_features[similarity_features]
        X = X.drop_duplicates()
        y = ['' for _ in range(X.shape[0])]  # dummy value since no gold standard is given at this point

        # predict action categories
        clf = ActionClassifier()
        clf.fit(X, y)
        pred = clf.predict(X)
        X['action:category'] = pred

        # add the predicted categories to the features
        df_categories = pd.get_dummies(X['action:category'], prefix='feature:action:category', prefix_sep=':') # 1-hot
        X[df_categories.columns] = df_categories
        self.df_features = self.df_features.merge(X, how='left')
        self.df['action:category'] = self.df_features['action:category']
        self.df_features.drop(columns=['action:category'], inplace=True)

    def _extract_previous(self) -> None:
        """
        Extract one-hot encoded features for previous category.

        :return: None
        """
        print('Extracting previous category ...')
        # initialize the columns
        for label in ANNOTATION_LABELS:
            self.df_features['feature:previous:'+label] = 0

        # add shifted values (i.e. previous event attributes) to df
        df_prev = self.df[[DEFAULT_CASE_KEY, 'action:category']].shift(1, fill_value=None)
        df_prev.loc[self.df[DEFAULT_CASE_KEY] != df_prev[DEFAULT_CASE_KEY], 'action:category'] = None
        # one hot encode
        df_prev = pd.get_dummies(df_prev['action:category'], prefix='feature:previous', prefix_sep=':')
        self.df_features[df_prev.columns] = df_prev

    def _extract_followed_by(self) -> None:
        """
        Extract one-hot encoded features for followed_by category.

        :return: None
        """
        print('Extracting following category ...')
        # initialize the columns
        for label in ANNOTATION_LABELS:
            self.df_features['feature:followed_by:'+label] = 0

        # add shifted values (i.e. following event attributes) to df
        df_follow = self.df[[DEFAULT_CASE_KEY, 'action:category']].shift(-1, fill_value=None)
        df_follow.loc[self.df[DEFAULT_CASE_KEY] != df_follow[DEFAULT_CASE_KEY], 'action:category'] = None
        # one hot encode
        df_follow = pd.get_dummies(df_follow['action:category'], prefix='feature:previous', prefix_sep=':')
        self.df_features[df_follow.columns] = df_follow

    def _extract_has_new_object(self):
        """
        Extract boolean value whether event contains a new object
        :return: None
        """
        print('Extracting has new objects ...')
        # initialize columns for object remains
        self.df_features.loc[:, 'feature:object:new'] = 0

        # get the previous objects for each event
        df_prev_obj = self.df[[DEFAULT_CASE_KEY, 'object:name']].copy()
        df_prev_obj['object:name:prev'] = self.df[[DEFAULT_CASE_KEY, 'object:name']].groupby([DEFAULT_CASE_KEY])\
            .apply(lambda x: (x+';').cumsum())['object:name']
        df_prev_obj[[DEFAULT_CASE_KEY, 'object:name:prev']] = df_prev_obj[[DEFAULT_CASE_KEY, 'object:name:prev']]\
            .shift(1, fill_value='')
        df_prev_obj.loc[self.df[DEFAULT_CASE_KEY] != df_prev_obj[DEFAULT_CASE_KEY], 'object:name:prev'] = ''

        # check if there are any new objects in an event
        def _check_new(x):
            current_obj = x['object:name'].split(';')
            prev_obj = x['object:name:prev'].split(';')
            new_obj = False
            for obj in current_obj:
                if obj not in prev_obj:
                    new_obj = True
                    break
            return new_obj
        df_new = df_prev_obj.apply(lambda x: _check_new(x), axis=1)
        self.df_features.loc[df_new, 'feature:object:new'] = 1

    def _extract_avg_vectors(self):
        lookup = {}

        def _get_avg_vector(phrases, prefix):
            phrases = re.sub('[^a-zA-Z, ]', '', phrases)
            cached = lookup.get(phrases, None)
            if cached is not None:
                return cached
            vectors = []
            for word in phrases.split(','):
                vec = self.embeddings.get(word)
                if vec is not None:
                    vectors.append(vec)
            avg_vector = pd.Series([np.NAN for _ in range(50)])
            if len(vectors) > 0:
                avg_vector = pd.Series(np.average(vectors, axis=0))
            avg_vector = avg_vector.rename(index=lambda dim: 'feature:embedding:{0}:dim{1}'.format(prefix, dim))
            lookup[phrases] = avg_vector
            return avg_vector

        for sem_role in ['action:name', 'object:name']:
            lookup = {}  # reset lookup
            tqdm.pandas(desc='Extract Average Embedding {0}'.format(sem_role))
            df_avg_vector = self.df[sem_role].progress_apply(lambda c: _get_avg_vector(c, sem_role))
            self.df_features = pd.concat([self.df_features, df_avg_vector], axis=1)

    def _extract_position(self):
        """
        Extract boolean value whether event is first or last in case
        :return: None
        """
        print('Extracting position + first/last ...')
        self.df_features['feature:position:order'] = self.df.groupby([DEFAULT_CASE_KEY]).cumcount()
        self.df_features['feature:position:first'] = 0
        self.df_features['feature:position:last'] = 0

        # add shifted values to get previous case
        df_prev = self.df[[DEFAULT_CASE_KEY]].shift(1, fill_value=None)
        self.df_features.loc[self.df[DEFAULT_CASE_KEY] != df_prev[DEFAULT_CASE_KEY], 'feature:position:first'] = 1
        # add shifted values to get next case
        df_next = self.df[[DEFAULT_CASE_KEY]].shift(-1, fill_value=None)
        self.df_features.loc[self.df[DEFAULT_CASE_KEY] != df_next[DEFAULT_CASE_KEY], 'feature:position:last'] = 1

    def _extract_object_remains(self):
        """
        Extract boolean value whether object remains in rest of case
        :return: None
        """
        print('Extracting objects remain ...')
        # initialize columns for object remains
        self.df_features.loc[:, 'feature:object:remains'] = 0

        # get the objects which are in the events after current event
        df_next_obj = self.df[[DEFAULT_CASE_KEY, 'object:name']][::-1].copy()
        df_next_obj['object:name:next'] = df_next_obj[[DEFAULT_CASE_KEY, 'object:name']].groupby([DEFAULT_CASE_KEY])\
            .apply(lambda x: (x+';').cumsum())['object:name']
        df_next_obj[[DEFAULT_CASE_KEY, 'object:name:next']] = df_next_obj[[DEFAULT_CASE_KEY, 'object:name:next']]\
            .shift(1, fill_value='')
        df_next_obj.loc[self.df[DEFAULT_CASE_KEY][::-1] != df_next_obj[DEFAULT_CASE_KEY], 'object:name:next'] = ''
        df_next_obj = df_next_obj[::-1]

        # check if all objects still remain after current event
        def _check_remains(x):
            current_obj = x['object:name'].split(';')
            next_obj = x['object:name:next'].split(';')
            obj_missing = False
            for obj in current_obj:
                if obj not in next_obj:
                    obj_missing = True
                    break
            return not obj_missing
        df_remains = df_next_obj.apply(lambda x: _check_remains(x), axis=1)
        self.df_features.loc[df_remains, 'feature:object:remains'] = 1

    def _extract_change(self):
        """
        Extract boolean values whether resource type or actor instance change from previous to current event
        :return: None
        """
        print('Extracting resource/actor change ...')
        prefix = 'feature:change:'
        # initialize the columns for each feature
        self.df_features[prefix + 'resource:type'] = 0
        self.df_features[prefix + 'org:actor:name'] = 0

        # add shifted values to get previous resource/actor
        df_prev = self.df[[DEFAULT_CASE_KEY, 'resource:type', 'org:actor:name']].shift(1, fill_value=None)
        self.df_features.loc[(self.df[DEFAULT_CASE_KEY] == df_prev[DEFAULT_CASE_KEY]) &
                             (self.df['resource:type'] != df_prev['resource:type']), prefix + 'resource:type'] = 1
        self.df_features.loc[(self.df[DEFAULT_CASE_KEY] == df_prev[DEFAULT_CASE_KEY]) &
                             (self.df['org:actor:name'] != df_prev['org:actor:name']), prefix + 'org:actor:name'] = 1

    def _extract_resource_type(self):
        """
        Extract boolean values whether resource type is human or system
        :return: None
        """
        print('Extracting resource type ...')
        self.df_features['feature:resource:system'] = 0
        self.df_features.loc[self.df['resource:type'] == 'sys', 'feature:resource:system'] = 1

        self.df_features['feature:resource:human'] = 0
        self.df_features.loc[self.df['resource:type'] == 'hum', 'feature:resource:human'] = 1

    def _extract_has_semantic_role(self):
        """
        Extract boolean values whether a semantic role is present in event (e.g. has objects, actors)
        :return: None
        """
        print('Extracting has semantic role ...')
        for role in SEMANTIC_ROLES:
            self.df_features['feature:has:' + role] = 0
            self.df_features.loc[self.df[role] != '', 'feature:has:' + role] = 1

    def _extract_amount_objects(self):
        """
        Extract values how many objects are in current and previous event
        :return: None
        """
        tqdm.pandas(desc='Extract object amounts')
        self.df_features['feature:amount:objects:now'] = \
            self.df[['object:name']].progress_apply(lambda x: len(x['object:name'].split(';')), axis=1)

        self.df_features['feature:amount:objects:before'] = \
            self.df_features[['feature:amount:objects:now']].shift(1, fill_value=0)
        self.df_features.loc[self.df_features['feature:position:first'] == 1, 'feature:amount:objects:before'] = 0

    def extract_all_features(self):
        self._extract_similarities()
        self._extract_action_category()
        self._extract_previous()
        self._extract_followed_by()
        self._extract_avg_vectors()
        self._extract_position()
        self._extract_has_new_object()
        self._extract_object_remains()
        self._extract_change()
        self._extract_resource_type()
        self._extract_has_semantic_role()
        self._extract_amount_objects()

        self.df_features = self.df_features.drop(columns=[DEFAULT_CASE_KEY]).sort_values(by=DEFAULT_ACTIVITY_KEY)
        self.df_features = self.df_features.groupby([EVENT_LOG_NAME_KEY, DEFAULT_ACTIVITY_KEY]).mean().reset_index()


if __name__ == '__main__':
    # extract features
    embeddings = EmbeddingsLoader('glove-wiki-gigaword-50')
    for filename in ALL_EVENT_LOG_NAMES:
        print('Extracting features for "{0}"'.format(filename))
        extractor = Extractor(filename, embeddings)
        extractor.extract_all_features()
        extractor.df_features.to_csv(os.path.join(FILEPATH_FEATURES, filename + '.csv'), index=False)

    # combine files into one file
    print('\nCombining all files ...')
    df_all = None
    for i, filename in enumerate(ALL_EVENT_LOG_NAMES):
        df_single = pd.read_csv(os.path.join(FILEPATH_FEATURES, filename + '.csv'), low_memory=False)
        if 'case:concept:name' in df_single.columns:
            print(filename)
        if i == 0:
            df_all = df_single
        else:
            df_all = pd.concat([df_all, df_single])

    print('\nSave final feature file ...')
    df_all.to_csv(os.path.join(FILEPATH_FEATURES, 'all.csv'), index=False)
