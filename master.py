# Importing packages used in the project
import pandas as pd 
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import itertools
import os

# Deep learning: 
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Embedding, concatenate, Dropout, concatenate

# Importing custom utility classes
from utiliy import Embeddings, clean_tweets, string_to_tensor, additional_features

# Reading the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Defining the path for word embedding
embedding_path = 'c:\\Users\\elaso\\embeddings\\glove.twitter.27B.200d.txt'
embedding_dim = 200

# Preprocecing the text
tweets = [clean_tweets(tweet) for tweet in train['text'].tolist()]

# Tokenizing the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

# Creating the embedding matrix
embedding = Embeddings(embedding_path, embedding_dim)
embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

# Getting the longest tweet
max_len = np.max([len(tweet.split()) for tweet in tweets])

# Creating the padded input for the deep learning model
X_train = string_to_tensor(tweets, tokenizer, max_len)

# Creating a separate feature matrix regarding length of tweet, capital words, etc.
X_additional = additional_features(tweets)

# Defining the deep learning model 
# The RNN part
inp1 = Input(shape=(max_len,))
x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
x = LSTM(256)(x)
x = Dropout(0.1)(x)
x = Dense(128, activation="relu")(x)
x = Model(inputs=inp1, outputs=x)

# Additional features
inp2 = Input(shape=(X_additional.shape[1], ))
y = Dense(256, activation='relu')(inp2)
y = Dense(128, activation='relu')(y)
y = Model(inputs=inp2, outputs=y)

# Combining the output of the two branches
concat = concatenate([x.output, y.output])

# Adding final layers
z = Dense(64, activation="relu")(concat)
z = Dense(1, activation="sigmoid")(z)   

# Compiling the final model
model = Model(inputs=[x.input, y.input], outputs=z)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

# Fitting the model
model_fited = model.fit(
    [X_train, X_additional],
    train['target'].values, 
    batch_size=100, 
    epochs=10
)  

# Preparing the test data 
tweets_test = [clean_tweets(tweet) for tweet in test['text'].tolist()]
X_test = string_to_tensor(tweets_test, tokenizer, max_len)
X_test_additional = additional_features(tweets_test)

# Making predictions
probs = [x[0] for x in model.predict([X_test, X_test_additional]).tolist()]
test['prob_is_genuine'] = probs
test['target'] = [1 if x > 0.5 else 0 for x in probs]
 
# Saving the submission to Kaggle to csv
if os.path.isdir('output'):
    test[['id', 'target']].to_csv('output/submission.csv', index=False)
