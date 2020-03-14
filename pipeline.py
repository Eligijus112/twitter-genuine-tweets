import numpy as np

# The main model class
from RNN_model import RnnModel

# Importing the word preprocesing class
from text_preprocessing import TextToTensor, clean_text

# Importing the word embedding class
from embeddings import Embeddings

# Loading the word tokenizer
from keras.preprocessing.text import Tokenizer

# For accuracy calculations
from sklearn.metrics import accuracy_score, f1_score


class Pipeline:
    """
    A class for the machine learning pipeline
    """
    def __init__(
        self, 
        X_train: list, 
        Y_train: list, 
        embed_path: str, 
        embed_dim: int,
        stop_words=[],
        X_test=[], 
        Y_test=[],
        max_len=None,
        epochs=3,
        batch_size=256
        ):

        # Preprocecing the text
        X_train = [clean_text(text, stop_words=stop_words) for text in X_train]
        Y_train = np.asarray(Y_train)
        
        # Tokenizing the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        # Saving the tokenizer
        self.tokenizer = tokenizer

        # Creating the embedding matrix
        embedding = Embeddings(embed_path, embed_dim)
        embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

        # Creating the padded input for the deep learning model
        if max_len is None:
            max_len = np.max([len(text.split()) for text in X_train])
        TextToTensor_instance = TextToTensor(
            tokenizer=tokenizer, 
            max_len=max_len
            )
        X_train = TextToTensor_instance.string_to_tensor(X_train)

        # Creating the model
        rnn = RnnModel(
            embedding_matrix=embedding_matrix, 
            embedding_dim=embed_dim, 
            max_len=max_len
        )
        rnn.model.fit(
            X_train,
            Y_train, 
            batch_size=batch_size, 
            epochs=epochs
        )

        self.model = rnn.model

        # If X_test is provided we make predictions with the created model
        if len(X_test)>0:
            X_test = [clean_text(text) for text in X_test]
            X_test = TextToTensor_instance.string_to_tensor(X_test)
            yhat = [x[0] for x in rnn.model.predict(X_test).tolist()]
            
            self.yhat = yhat

            # If true labels are provided we calculate the accuracy of the model
            if len(Y_test)>0:
                self.acc = accuracy_score(Y_test, [1 if x > 0.5 else 0 for x in yhat])
                self.f1 = f1_score(Y_test, [1 if x > 0.5 else 0 for x in yhat])