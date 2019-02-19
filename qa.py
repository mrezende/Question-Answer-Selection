import numpy as np
from model import QAModel
from data import QAData
import random
from scipy.stats import rankdata
import json
from keras.preprocessing.text import Tokenizer
import argparse
import logging
import os
import sys
import pandas as pd
from ggplot import *
from keras import backend as K

def train(train_model, model_name='baseline', epochs=10, batch_size=64, validation_split=0.2):
    # load training data
    qa_data = QAData()
    questions, good_answers, bad_answers = qa_data.get_training_data()

    logger.info(f'Training: epochs {epochs}, batch_size {batch_size}, validation_split {validation_split}')
    # train the model
    Y = np.zeros(shape=(questions.shape[0],))
    callback = train_model.fit(
        [questions, good_answers, bad_answers],
        Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    df = pd.DataFrame(callback.history)
    df.insert(0, 'epochs', range(0, len(df)))
    df = pd.melt(df, id_vars=['epochs'])
    plot = ggplot(aes(x='epochs', y='value', color='variable'), data=df) + geom_line()
    filename = f'{model_name}_plot.png'
    logger.info(f'saving loss, val_loss plot: {filename}')
    plot.save('baseline.png')

    # save the trained model
    train_model.save_weights(f'model/train_weights_{model_name}.h5', overwrite=True)
    logger.info(f'model/train_weights_{model_name}.h5')
    K.clear_session()


def main(mode='train', question=None, answers=None, epochs=100, batch_size=64, validation_split=0.2):
    """
    This function is used to train, predict or test

    Args:
        mode (str): train/preddict/test
        question (str): this contains the question
        answers (list): this contains list of answers in string format

    Returns:
        index (integer): index of the most likely answer
    """

    # get the train and predict model model


    samples = []
    with open('data/samples_for_tokenizer.json', 'r') as read_file:
        samples = json.load(read_file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(samples)

    embedding_file = "./data/word2vec_100_dim.embeddings"

    qa_model = QAModel()
    train_model, predict_model = qa_model.get_lstm_cnn_model(embedding_file, len(tokenizer.word_index) + 1)

    if mode == 'train':
        logger.info('Default training: Baseline')
        logger.info('enc_timesteps = 30,\
                           dec_timesteps = 30, hidden_dim = 50, kernel_size = 100, filters = [1]')

        train_model.summary()
        train(train_model, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        # small model
        enc_timesteps = 30
        dec_timesteps = 30
        hidden_dim = 10
        kernel_size = 20
        filters = [1]
        small_train_model, small_predict_model = qa_model.get_lstm_cnn_model(embedding_file,
                                                                             len(tokenizer.word_index) + 1,
                                                                             enc_timesteps=enc_timesteps,
                                                                             dec_timesteps=dec_timesteps,
                                                                             kernel_size=kernel_size,
                                                                             hidden_dim=hidden_dim,
                                                                             filters=filters)
        small_train_model.summary()
        logger.info('Model training: Small')
        logger.info(f'enc_timesteps = {enc_timesteps},\
                                   dec_timesteps = {dec_timesteps},'
                    f' hidden_dim = {hidden_dim}, kernel_size = {kernel_size}, '
                    f'filters = {filters}')
        train(small_train_model, model_name='small', epochs=epochs,
                    batch_size=batch_size, validation_split=validation_split)


        # larger model

        enc_timesteps = 30
        dec_timesteps = 30
        hidden_dim = 200
        kernel_size = 500
        filters = [1]
        larger_train_model, larger_predict_model = qa_model.get_lstm_cnn_model(embedding_file,
                                                                             len(tokenizer.word_index) + 1,
                                                                             enc_timesteps=enc_timesteps,
                                                                             dec_timesteps=dec_timesteps,
                                                                             kernel_size=kernel_size,
                                                                             hidden_dim=hidden_dim,
                                                                             filters=filters)
        larger_train_model.summary()
        logger.info('Model training: Larger')
        logger.info(f'enc_timesteps = {enc_timesteps},\
                                           dec_timesteps = {dec_timesteps},'
                    f' hidden_dim = {hidden_dim}, kernel_size = {kernel_size}, '
                    f'filters = {filters}')
        train(larger_train_model, model_name='larger', epochs=epochs,
                    batch_size=batch_size, validation_split=validation_split)

    elif mode == 'predict':
        # load the evaluation data
        data = []
        with open('data/test.json') as read_file:
            data = json.load(read_file)

        random.shuffle(data)

        # load weights from trained model
        qa_data = QAData()
        predict_model.load_weights('model/predict_weights_epoch_10.h5')

        c = 0
        c1 = 0
        for i, d in enumerate(data):
            print (i, len(data))

            # pad the data and get it in desired format
            answers, question = qa_data.process_data(d)

            # get the similarity score
            sims = predict_model.predict([question, answers])

            n_good = len(d['good'])
            max_r = np.argmax(sims)
            max_n = np.argmax(sims[:n_good])
            r = rankdata(sims, method='max')
            c += 1 if max_r == max_n else 0
            c1 += 1 / float(r[max_r] - r[max_n] + 1)

        precision = c / float(len(data))
        mrr = c1 / float(len(data))
        print ("Precision", precision)
        print ("MRR", mrr)
    elif mode == 'test':
        # question and answers come from params
        qa_data = QAData()
        answers, question = qa_data.process_test_data(question, answers)

        # load weights from the trained model
        predict_model.load_weights('model/predict_weights_epoch_10.h5')

        # get similarity score
        sims = predict_model.predict([question, answers])
        max_r = np.argmax(sims)
        return max_r

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='run question answer selection')
    parser.add_argument('--mode', metavar='MODE', type=str, default="train", help='mode: train/predict/test')
    parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=100, help='epochs for train')
    parser.add_argument('--batch_size', metavar='BATCH SIZE', type=int, default=64, help='batch size for train')
    parser.add_argument('--validation_split', metavar='VALIDATION SPLIT', type=float, default=0.2,
                        help='validation split: 0.1 for an example')

    args = parser.parse_args()

    # configure logging
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    main(mode=args.mode, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split)

def test(question, answers):
    return main(mode='test', question=question, answers=answers)


