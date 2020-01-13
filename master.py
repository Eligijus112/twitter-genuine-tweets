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
from RNN_model import RnnModel

# Importing custom utility classes
from utiliy import Embeddings, clean_tweets, string_to_tensor, additional_features

# K fold validation package
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Reading the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Defining the path for word embedding
embedding_path = 'c:\\Users\\elaso\\embeddings\\glove.twitter.27B.200d.txt'
embedding_dim = 200

# Preprocecing the text
tweets = [clean_tweets(tweet) for tweet in train['text'].tolist()]
Y = np.asarray(train['target'].tolist())

# Splitting the tweets into a train and test sets for k fold analysis
kfold = KFold(n_splits=5)
acc = []
for train_index, test_index in kfold.split(tweets):
    tweets_train = np.array(tweets)[train_index]
    tweets_test = np.array(tweets)[test_index]

    # Tokenizing the tweets
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets_train)

    # Creating the embedding matrix
    embedding = Embeddings(embedding_path, embedding_dim)
    embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

    # Getting the longest tweet
    max_len = np.max([len(tweet.split()) for tweet in tweets_train])

    # Creating the padded input for the deep learning model
    X_train = string_to_tensor(tweets_train, tokenizer, max_len)

    # Creating a separate feature matrix regarding length of tweet, capital words, etc.
    X_additional = additional_features(tweets_train)

    # Creating an RNN model object
    rnn = RnnModel(
        embedding_matrix=embedding_matrix, 
        embedding_dim=embedding_dim, 
        max_len=max_len, 
        X_additional=pd.DataFrame({})
        )
    model = rnn.return_model()
    hyper = rnn.hyperparameters()    

    # Preparing the test data 
    tweets_test = [clean_tweets(tweet) for tweet in tweets_test]
    X_test = string_to_tensor(tweets_test, tokenizer, max_len)
    X_test_additional = additional_features(tweets_test)

    # Training the model with a validation set
    if type(model.input) == list:
        model_fited = model.fit(
            [X_train, X_additional],
            Y[train_index], 
            batch_size=hyper['batch_size'], 
            epochs=hyper['epochs'],
            validation_data = ([X_test, X_test_additional], Y[test_index])
        ) 
        probs = [x[0] for x in model.predict([X_test, X_test_additional]).tolist()]
    else:
        model_fited = model.fit(
            X_train,
            Y[train_index], 
            batch_size=hyper['batch_size'], 
            epochs=hyper['epochs'],
            validation_data = (X_test, Y[test_index])
        )   
        probs = [x[0] for x in model.predict(X_test).tolist()]
    
    # Calculating the accuracy
    forecasts = [1 if x > hyper['p_tresh'] else 0 for x in probs]
    acc += [accuracy_score(Y[test_index], forecasts)]
    print(f'The accuracy score is: {acc[-1]}') 

print(f'Total mean accuracy is: {np.mean(acc)}')       

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

# Preparing the test data 
tweets_test = [clean_tweets(tweet) for tweet in test['text'].tolist()]
X_test = string_to_tensor(tweets_test, tokenizer, max_len)
X_test_additional = additional_features(tweets_test)

# Defining the deep learning model 
rnn = RnnModel(
        embedding_matrix=embedding_matrix, 
        embedding_dim=embedding_dim, 
        max_len=max_len, 
        X_additional=pd.DataFrame({})
        )
model = rnn.return_model()
hyper = rnn.hyperparameters()

# Fitting the model
if type(model.input) == list:
    model_fited = model.fit(
        [X_train, X_additional],
        Y, 
        batch_size=hyper['batch_size'], 
        epochs=hyper['epochs']
    ) 
    probs = [x[0] for x in model.predict([X_test, X_test_additional]).tolist()]
else:
    model_fited = model.fit(
        X_train,
        Y, 
        batch_size=hyper['batch_size'], 
        epochs=hyper['epochs']
    )   
    probs = [x[0] for x in model.predict(X_test).tolist()]
    
# Making predictions
test['prob_is_genuine'] = probs
test['target'] = [1 if x > 0.5 else 0 for x in probs]
 
# Saving the submission to Kaggle to csv
if os.path.isdir('output'):
    test[['id', 'target']].to_csv('output/submission.csv', index=False)
