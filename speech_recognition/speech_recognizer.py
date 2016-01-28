import os
import sys
import numpy
import scipy.io.wavfile as wavfile

from hmm import HMM
from third_party import mfcc


class SpeechRecognizer():
    def __init__(self, dictionary_path):
        self.dictionary = {}
        self.hmms = []
        self.rate = None
        self.__load_dictionary(dictionary_path)

    # loads all wav files from given dictionary
    # computes coefficients using MFCC
    # trains recognizer using HMMs
    def __load_dictionary(self, dir_name):
        print('Loading dictionary from ' + dir_name + "...")
        for dir in os.walk(dir_name).next()[1]:
            self.dictionary[dir] = []
            for file in os.walk(dir_name + '/' + dir).next()[2]:
                if file.endswith('.wav'):
                    rate, data = wavfile.read(dir_name + '/' + dir + '/' + file)
                    if self.rate is not None and self.rate != rate:
                        print('Error: Dictionary sampling rate not constant.')
                    self.rate = rate
                    coefficients = self.__get_features(data)
                    self.dictionary[dir].append((coefficients, len(data)))
        print('Done.')
        print('Computing HMMs...')
        for key, value in self.dictionary.items():
            hmm = HMM(key)
            model_size = self.__get_model_size([len(x[0]) for x in value])
            hmm.train(model_size, [x[0] for x in value])
            self.hmms.append(hmm)
            print('Trained {0} with {1} states'.format(key, model_size))
        print('Done.')

    # helper for calculating number of HMM states
    def __get_model_size(self, training_set_lengths):
        avg_len = numpy.average(training_set_lengths)
        model_size = int(avg_len / 10.0)
        return model_size

    # calculates mfcc for given speech
    def __get_features(self, data):
        # TODO: remove third party dependency
        return mfcc(data, self.rate)

    # search through HMMs for one with highest likelihood
    def search_word(self, data):
        coefficients = self.__get_features(data)
        likelihood = -sys.float_info.max
        best_word = ''
        for hmm in self.hmms:
            e = hmm.match_viterbi(coefficients)
            if e > likelihood:
                likelihood = e
                best_word = hmm.word
        return best_word, likelihood
