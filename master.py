# Importing packages used in the project
import pandas as pd 
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import itertools
import os

# Deep learning: 
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Embedding, concatenate, Dropout

# Importing custom utility classes
from utiliy import Embeddings, clean_tweets

# Reading the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Defining the path for word embedding
embedding_path = 'c:\\Users\\elaso\\embeddings\\glove.twitter.27B.200d.txt'
embedding_dim = 200

# Preprocecing the text
tweets = train['text'].tolist()
tweets = [clean_tweets(tweet) for tweet in tweets]

# Tokenizing the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

# Creating the embedding matrix
embedding = Embeddings(embedding_path, embedding_dim)
embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

# Getting the longest tweet
max_len = np.max([len(tweet.split()) for tweet in tweets])

# Creating the padded input for the deep learning model
X_train = tokenizer.texts_to_sequences(tweets)
X_train = pad_sequences(X_train, maxlen=max_len)

# Defining the deep learning model
inp = Input(shape=(max_len,))
x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp)
x = LSTM(400, return_sequences=True)(x)
x = LSTM(200)(x)
x = Dropout(0.15)(x)
x = Dense(64, activation="relu")(x)
outp = Dense(1, activation="sigmoid")(x)   

model = Model(inputs=inp, outputs=outp)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

# Fitting the model
model_fited = model.fit(
    X_train,
    train['target'].values, 
    batch_size=100, 
    epochs=10
)  

# Preparing the test data 
tweets_test = test['text'].tolist()
tweets_test = [clean_tweets(tweet) for tweet in tweets_test]

X_test = tokenizer.texts_to_sequences(tweets_test)
X_test = pad_sequences(X_test, maxlen=max_len)

# Making predictions
probs = [x[0] for x in model.predict(X_test).tolist()]

# Creating the sample submission
submission = pd.DataFrame({
    'id': test['id'], 
    'target': [1 if x > 0.5 else 0 for x in probs]
})

# Saving to csv
if os.path.isdir('output'):
    submission.to_csv('output/submission.csv', index=False)
