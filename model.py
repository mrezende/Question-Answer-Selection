from keras import backend as K
from keras.layers import Embedding
from keras.layers import LSTM, Input, concatenate, Lambda, Dot
from keras.layers.wrappers import Bidirectional
from keras.layers import Conv1D
from keras.models import Model
import numpy as np

class QAModel():
    def get_cosine_similarity(self):
        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())


    def get_bilstm_model(self, embedding_file, vocab_size, enc_timesteps = 30, dec_timesteps = 30, hidden_dim = 5):
        """
        Return the bilstm training and prediction model

        Args:
            embedding_file (str): embedding file name
            vacab_size (integer): size of the vocabulary

        Returns:
            training_model: model used to train using cosine similarity loss
            prediction_model: model used to predict the similarity
        """

        margin = 0.05


        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')
        answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

        weights = np.load(embedding_file)
        qa_embedding = Embedding(input_dim=vocab_size, output_dim=weights.shape[1], mask_zero=True, weights=[weights])
        bi_lstm = Bidirectional(LSTM(activation='tanh', dropout=0.2, units=hidden_dim, return_sequences=False))

        # embed the question and pass it through bilstm
        question_embedding =  qa_embedding(question)
        question_enc_1 = bi_lstm(question_embedding)

        # embed the answer and pass it through bilstm
        answer_embedding =  qa_embedding(answer)
        answer_enc_1 = bi_lstm(answer_embedding)

        # get the cosine similarity
        similarity = self.get_cosine_similarity()

        #question_answer_merged = merge(inputs=[question_enc_1, answer_enc_1], mode=similarity, output_shape=lambda _: (None, 1))
        question_answer_merged = Dot(axes=-1, normalize=True)([question_enc_1, answer_enc_1])
        #question_answer_concatenated = concatenate([question_enc_1, answer_enc_1])
        #question_answer_merged = Lambda(function=similarity, output_shape=lambda _: (None, 1))(
        #    question_answer_concatenated)

        lstm_model = Model(name="bi_lstm", inputs=[question, answer], outputs=question_answer_merged)
        good_similarity = lstm_model([question, answer_good])
        bad_similarity = lstm_model([question, answer_bad])

        # compute the loss
        # loss = merge(
        #    [good_similarity, bad_similarity],
        #    mode=lambda x: K.relu(margin - x[0] + x[1]),
        #    output_shape=lambda x: x[0]
        #)

        good_bad_concatenated = concatenate([good_similarity, bad_similarity])
        loss = Lambda(function=lambda x: K.relu(margin - x[0] + x[1]), output_shape=lambda x: x[0])(
            good_bad_concatenated)

        # return model
        model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
        model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")


        return model


    def get_lstm_cnn_model(self,embedding_file,  vocab_size, enc_timesteps = 30,
                           dec_timesteps = 30, hidden_dim = 50, kernel_size = 100, filters = [1]):
        """
        Return the bilstm + cnn training and prediction model

        Args:
            embedding_file (str): embedding file name
            vacab_size (integer): size of the vocabulary

        Returns:
            training_model: model used to train using cosine similarity loss
            prediction_model: model used to predict the similarity
        """

        margin = 0.05

        weights = np.load(embedding_file)

        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

        # embed the question and answers
        qa_embedding = Embedding(input_dim=vocab_size, output_dim=weights.shape[1], weights=[weights])
        question_embedding =  qa_embedding(question)
        answer_embedding =  qa_embedding(answer)

        # pass the question embedding through bi-lstm
        f_rnn = LSTM(hidden_dim, return_sequences=True)
        b_rnn = LSTM(hidden_dim, return_sequences=True)
        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        #question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)
        question_pool = concatenate([qf_rnn, qb_rnn], axis=-1)

        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)

        #answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)
        answer_pool = concatenate([af_rnn, ab_rnn], axis=-1)

        # pass the embedding from bi-lstm through cnn
        #cnns = [Convolution1D(filter_length=filter_length,nb_filter=500,activation='tanh',border_mode='same') for filter_length in [1, 2, 3, 5]]
        cnns = [Conv1D(filters=filter_length, kernel_size=kernel_size, activation='tanh', padding='same') for filter_length in filters]

        #question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
        question_cnn = None
        if len(cnns) > 1:
            question_cnn = concatenate([cnn(question_pool) for cnn in cnns])
        else:
            question_cnn = cnns[0](question_pool)
        
        
            

        #answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')
        answer_cnn = None
        if len(cnns) > 1:
            answer_cnn = concatenate([cnn(answer_pool) for cnn in cnns])
        else:
            answer_cnn = cnns[0](answer_pool)
        
            

        # apply max pooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        # get similarity similarity score
        similarity = self.get_cosine_similarity()
        #merged_model = merge([question_pool, answer_pool],mode=similarity, output_shape=lambda _: (None, 1))
        #question_answer_concatenated = concatenate([question_pool, answer_pool])
        merged_model = Dot(axes=-1, normalize=True)([question_pool, answer_pool])
        #merged_model = Lambda(function=similarity, output_shape=lambda _: (None, 1))([question_pool, answer_pool])

        lstm_convolution_model = Model(inputs=[question, answer], outputs=merged_model, name='lstm_convolution_model')
        good_similarity = lstm_convolution_model([question, answer_good])
        bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
        #loss = merge(
        #    [good_similarity, bad_similarity],
        #    mode=lambda x: K.relu(margin - x[0] + x[1]),
        #    output_shape=lambda x: x[0]
        #)

        #good_bad_concatenated = concatenate([good_similarity, bad_similarity])
        loss = Lambda(function=lambda x: K.relu(margin - x[0] + x[1]), output_shape=lambda x: x[0])(
            [good_similarity, bad_similarity])

        model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='model')
        model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        return model
