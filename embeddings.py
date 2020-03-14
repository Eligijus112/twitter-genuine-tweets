import numpy as np


class Embeddings():
    """
    A class to read the word embedding file and to create the word embedding matrix
    """

    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    def create_embedding_matrix(self, tokenizer=None, max_features=None):
        """
        A method to create the embedding matrix
        """
        model_embed = self.get_embedding_index()

        if max_features is None:
            max_features = len(model_embed)

        word_index = model_embed
        if tokenizer is not None: 
            word_index = tokenizer.word_index

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for index, word in enumerate(word_index.keys()):
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        return embedding_matrix