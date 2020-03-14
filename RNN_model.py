# Deep learning: 
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Embedding, concatenate, Dropout, concatenate
from keras.layers import Bidirectional

class RnnModel():
    """
    A recurrent neural network for semantic analysis
    """

    def __init__(self, embedding_matrix, embedding_dim, max_len, X_additional=None):
        
        inp1 = Input(shape=(max_len,))
        x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64))(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(16, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)    
        model = Model(inputs=inp1, outputs=x)

        model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
        self.model = model
     