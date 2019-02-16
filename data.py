import random
from collections import namedtuple
import pickle
import json
from keras.preprocessing.text import Tokenizer

class QAData():
    """
    Load the train/predecit/test data
    """

    def __init__(self):
        self.samples = []
        with open('data/samples_for_tokenizer.json', 'r') as read_file:
            self.samples = json.load(read_file)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.samples)
        self.dec_timesteps=150
        self.enc_timesteps=150

        self.training_set = []
        with open('data/train.json', 'r') as read_file:
            self.training_set = json.load(read_file)

        self.answers = []
        with open('data/answers.json', 'r') as read_file:
            self.answers = json.load(read_file)

    def pad(self, data, length):
        """
        pad the data to meet given length requirement

        Args:
            data (vector): vector of question or answer
            length(integer): length of desired vector
        """

        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)

    def get_training_data(self):
        """
        Return training question and answers
        """

        questions = []
        good_answers = []
        for j, qa in enumerate(self.training_set):
            questions.extend([qa['question']] * len(qa['answers']))
            good_answers.extend([i for i in qa['answers']])

        # pad the question and answers
        questions = self.pad(questions, self.enc_timesteps)
        good_answers = self.pad(good_answers, self.dec_timesteps)
        bad_answers = self.pad(random.sample(self.answers, len(good_answers)), self.dec_timesteps)

        return questions,good_answers,bad_answers

    def process_data(self, d):
        """
        Process the predection data
        """

        answers = d['good'] + d['bad']
        answers = self.pad(answers, self.dec_timesteps)
        question = self.pad([d['question']] * len(answers), self.enc_timesteps)
        return answers,question

    def process_test_data(self, question, answers):
        """
        Process the test data
        """

        answer_unpadded = []
        for answer in answers:
            print (answer.split(' '))
            answer_unpadded.append([self.vocabulary[word] for word in answer.split(' ')])
        answers = self.pad(answer_unpadded, self.dec_timesteps)
        question = self.pad([[self.vocabulary[word] for word in question.split(' ')]] * len(answers), self.enc_timesteps)
        return answers, question
