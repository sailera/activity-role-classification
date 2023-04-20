import gensim.downloader as api
import numpy as np

from src.feature_extraction.load_reference_actions import load_reference_actions

REFERENCE_ACTIONS = load_reference_actions()


def cosine_similarity(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class EmbeddingsLoader:
    def __init__(self, embeddings_filename):
        self.filename = embeddings_filename
        self.embeddings_index = api.load(self.filename)
        self.reference_vectors = self._get_reference_vectors()

    def get(self, word):
        if self.embeddings_index.has_index_for(word):
            return self.embeddings_index[word]
        else:
            return None

    def _get_reference_vectors(self):
        reference_vectors = [(action, [self.get(action), *[self.get(ref) for ref in refs if self.get(ref) is not None]])
                             for (action, refs) in REFERENCE_ACTIONS]
        return reference_vectors


if __name__ == '__main__':
    glove_embeddings = EmbeddingsLoader('glove-wiki-gigaword-50')
    queen = glove_embeddings.get('queen')
    king = glove_embeddings.get('king')
    print(cosine_similarity(queen, king))
