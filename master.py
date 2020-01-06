# Importing packages used in the project
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from keras.preprocessing.text import Tokenizer

# Importing custom utility classes
from utiliy import Embeddings

# Reading the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Defining the path for word embedding
embedding_path = 'c:\\Users\\elaso\\embeddings\\glove.twitter.27B.200d.txt'
embedding_dim = 200

# Creating a class object of the embedding
embedding = Embeddings(embedding_path, embedding_dim)
embedding_dict = embedding.get_embedding_index()
