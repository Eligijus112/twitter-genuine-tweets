import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences


class Embeddings():

    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    def create_embedding_matrix(self, tokenizer, max_features):
        """
        A method to create the embedding matrix
        """
        model_embed = self.get_embedding_index()

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for word, index in tokenizer.word_index.items():
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        return embedding_matrix        

def clean_tweets(tweet):
    """
    A method to clean a tweet for special characters
    """
    tweet = re.sub('#', ' # ', tweet)
    tweet = re.sub('@', ' @ ', tweet)
    tweet = re.sub( r'\s+', ' ', tweet).strip()

    return tweet

def string_to_tensor(string_list, tokenizer, max_len):
    """
    A method to convert a string list to a tensor for a deep learning model
    """    
    string_list = tokenizer.texts_to_sequences(string_list)
    string_list = pad_sequences(string_list, maxlen=max_len)
    
    return string_list

def additional_features(string_list):
    """
    A method that creates an additional feature matrix regarding the tweets
    """    
    return pd.DataFrame({
        'length': [len(x) for x in string_list],
        'capital_letter_count': [len(re.findall(r'[A-Z]', x)) for x in string_list],
        'question_marks': [len(re.findall(r'\?', x)) for x in string_list],
        'exclamations': [len(re.findall(r'\!', x)) for x in string_list]
    })