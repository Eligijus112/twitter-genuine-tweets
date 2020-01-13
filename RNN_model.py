# Deep learning: 
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Embedding, concatenate, Dropout, concatenate
from keras.layers import Bidirectional

class RnnModel():

    def __init__(self, embedding_matrix, embedding_dim, max_len, X_additional=None):
        # The RNN part
        inp1 = Input(shape=(max_len,))
        x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(150))(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.1)(x)

        # Additional features
        if not X_additional.empty:
            inp2 = Input(shape=(X_additional.shape[1], ))
            y = Dense(256, activation='relu')(inp2)
            y = Dense(128, activation='relu')(y)
            y = Model(inputs=inp2, outputs=y)

            # Combining the output of the two branches
            x = Model(inputs=inp1, outputs=x)
            concat = concatenate([x.output, y.output])

            # Adding final layers
            z = Dense(128, activation="relu")(concat)
            z = Dense(64, activation="relu")(concat)
            z = Dense(1, activation="sigmoid")(z)   

            # Compiling the final model
            model = Model(inputs=[x.input, y.input], outputs=z)
        else:
            x = Dense(64, activation="relu")(x)
            x = Dense(1, activation="sigmoid")(x)    
            model = Model(inputs=inp1, outputs=x)

        model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
        self.model = model

    def return_model(self):
        return self.model 

    @staticmethod
    def hyperparameters():
        return {
            'epochs': 3,
            'batch_size': 100,
            'p_tresh': 0.5 
        }       