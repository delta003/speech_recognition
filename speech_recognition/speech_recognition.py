import os
import scipy.io.wavfile as wavfile

from word_hmm import WordHmm
from third_party.python_speech_features.base import mfcc

class SpeechRecognizer():
    def __init__(self):
        self.__dictionary = None
        self.__rate = None
        self.__hmms = None

    def have_dictionary(self):
        return self.__dictionary is not None

    def dictionary_sampling_rate(self):
        return self.__rate

    # loads all wav files from word_base/
    # returns list of pairs (string word, coefficients)
    def load_dictionary(self, dir_name):
        self.dictionary = {}
        print('Loading dictionary from ' + dir_name + "...")
        for dir in os.walk(dir_name).next()[1]:
            self.dictionary[dir] = []
            for file in os.walk(dir_name + '/' + dir).next()[2]:
                if file.endswith('.wav'):
                    rate, data = wavfile.read(dir_name + '/' + dir + '/' + file)
                    if self.rate != None and self.rate != rate:
                        print('Error: Dictionary sampling rate not constant.')
                    self.rate = rate
                    coefficients = self.__get_coefficients(data)
                    self.dictionary[dir].append(coefficients)
        print('Done.')
        self.__hmms = []
        print('Computing HMMs...')
        for key, value in self.dictionary.items():
            hmm = WordHmm(key, value)
            self.__hmms.append(hmm)
        print('Done.')


    # calculates coefficients for given speech in wav file,
    # returns list of vectors, each vector represents coefficients
    # for one interval
    def __get_coefficients(self, data):
        # TODO: remove third party mfcc dependency
        return mfcc(data. self.__rate)

    def search_word(self, rate, data):
        if rate != self.__rate:
            print('Word sampling rate do NOT match dictionary sampling rate')
            return '', -1.0
        coefficients = self.__get_coefficients(data)
        best_score = -1.0
        best_word = ''
        for hmm in self.__hmms:
            score = hmm.try_matching(coefficients)
            if score > best_score:
                best_score = score
                best_word = hmm.word()
        return best_word, best_score

