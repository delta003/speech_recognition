import numpy as np
import sys

from utility import chunks, mean_vector, covariance_matrix, \
    gaussian_log_likelihood, float_vector_equal


class HmmState:
    def __init__(self, mean, covariance, loop):
        self.mean = mean  # mean for Gaussian distribution
        self.covariance = covariance  # covariance matrix, just diagonal
        self.loop = loop  # probability to stay in this state

    # builds HmmState from list of vectors and number of training examples
    @staticmethod
    def build(vectors, n):
        mean = mean_vector(vectors)
        covariance = covariance_matrix(mean, vectors)
        loop = float(len(vectors) - n) / float(len(vectors))
        return HmmState(mean, covariance, loop)

    def __str__(self):
        n = len(self.mean)
        ret = '{0} '.format(n)
        for i in range(0, n):
            ret += '{0:.6f} '.format(self.mean[i])
        for i in range(0, n):
            ret += '{0:.6f} '.format(self.covariance[i][i])
        ret += '{0:.6f}'.format(self.loop)
        return ret

    # computes the output log likelihood for given vector x
    def output_likelihood(self, x):
        return gaussian_log_likelihood(x, self.mean, self.covariance)

    # computes log transition for staying in this state,
    # for transition 0.0 outputs big negative
    def log_stay(self):
        if self.loop == 0.0:
            return -1000000000.0
        else:
            return np.log(self.loop)

    # computes log transition for moving to next state,
    # for transition 1.0 outputs big negative
    def log_next(self):
        if self.loop == 1.0:
            return -1000000000.0
        else:
            return np.log(1.0 - self.loop)

    @staticmethod
    def deserialize(text):
        parts = text.split(' ')
        n = int(parts[0])
        mean = []
        for i in range(0, n):
            mean.append(float(parts[i + 1]))
        covariance = [[0.0 for i in range(0, n)] for j in range(0, n)]
        for i in range(0, n):
            covariance[i][i] = float(parts[i + n + 1])
        loop = float(parts[-1])
        return HmmState(mean, covariance, loop)


# Gaussian HMM
class HMM:
    def __init__(self, word):
        self.word = word
        self.states = []
        self.model_size = 0
        self.too_many_states = False

    # trains this HMM on given training set (list of lists of vectors)
    # updates self.states
    # The Segmental K-Means Algorithm + The Viterbi Algorithm
    def train(self, model_size, training_set):
        # dummy check
        # TODO: remove this
        for t in training_set:
            if len(t) < model_size:
                self.too_many_states = True
                print('ERROR IN TRAIN')
                return
        training_set_in_parts = []
        for element in training_set:
            training_set_in_parts.append(chunks(element, model_size))
        previous_states = self.__compute_states(training_set_in_parts,
                                                model_size)
        iteration = 0
        while True:
            iteration += 1
            new_training_set_in_parts = []
            for element in training_set_in_parts:
                joined = []
                for i in range(0, model_size):
                    joined += element[i]
                new_training_set_in_parts.append(self.__update_training_element(
                        previous_states, joined))
            new_states = self.__compute_states(training_set_in_parts,
                                               model_size)
            if self.__equal(training_set_in_parts, new_training_set_in_parts):
                self.states = new_states
                self.model_size = model_size
                break
            else:
                training_set_in_parts = new_training_set_in_parts

    # helper for training, compute states using each training element decomposed
    # to parts
    def __compute_states(self, training_set_in_parts, model_size):
        states = []
        for i in range(0, model_size):
            joined = []
            for j in range(0, len(training_set_in_parts)):
                joined += training_set_in_parts[j][i]
            states.append(HmmState.build(joined, len(training_set_in_parts)))
        return states

    # helper for training, checks if two decompositions are equal
    def __equal(self, decomp1, decomp2):
        assert len(decomp1) == len(decomp2)
        for i in range(0, len(decomp1)):
            assert len(decomp1[i]) == len(decomp2[i])
            for j in range(0, len(decomp1[i])):
                if len(decomp1[i][j]) != len(decomp2[i][j]):
                    return False
                # sanity check, can be removed
                for k in range(0, len(decomp1[i][j])):
                    if not float_vector_equal(decomp1[i][j][k],
                                              decomp2[i][j][k]):
                        return False
        return True

    # helper for training, given the states and element decomposed to parts,
    # updates decomposition using best Viterbi path through states
    # returns new decomposition
    def __update_training_element(self, states, elements):
        likelihoods = {(0, 0) : states[0].output_likelihood(elements[0])}
        comes_from = {}
        for i in range(1, len(elements)):
            likelihoods[(0, i)] = likelihoods[(0, i - 1)] + \
                                    states[0].log_stay() + \
                                    states[0].output_likelihood(elements[i])
            comes_from[(0, i)] = (0, i - 1)
        for i in range(1, len(states)):
            likelihoods[(i, i)] = likelihoods[(i - 1, i - 1)] + \
                                    states[i - 1].log_next() + \
                                    states[i].output_likelihood(elements[i])
            comes_from[(i, i)] = (i - 1, i - 1)
        for i in range(1, len(states)):
            for j in range(i + 1, len(elements)):
                left = likelihoods[(i, j - 1)] + \
                       states[i].log_stay() + \
                       states[i].output_likelihood(elements[j])
                down = likelihoods[(i - 1, j - 1)] + \
                       states[i - 1].log_next() + \
                       states[i].output_likelihood(elements[j])
                if left > down:
                    likelihoods[(i, j)] = left
                    comes_from[(i, j)] = (i, j - 1)
                else:
                    likelihoods[(i, j)] = down
                    comes_from[(i, j)] = (i - 1, j - 1)
        last = (len(states) - 1, len(elements) - 1)
        current_decomp = [elements[-1]]
        decomp = []
        while last != (0, 0):
            previous = comes_from[last]
            if previous[0] == last[0]:
                current_decomp.append(elements[previous[1]])
            else:
                decomp.append(current_decomp)
                current_decomp = [elements[previous[1]]]
            last = previous
        decomp.append(current_decomp)
        for d in decomp:
            d.reverse()
        decomp.reverse()
        return decomp

    # tries to match test with this HMM and returns likelihood
    # The Viterbi Algorithm
    def match_viterbi(self, test):
        # dummy check, just in case
        if self.too_many_states or len(test) < self.model_size:
            return -sys.float_info.max
        likelihoods = {(0, 0) : self.states[0].output_likelihood(test[0])}
        for i in range(1, len(test)):
            likelihoods[(0, i)] = likelihoods[(0, i - 1)] + \
                                    self.states[0].log_stay() + \
                                    self.states[0].output_likelihood(test[i])
        for i in range(1, self.model_size):
            likelihoods[(i, i)] = likelihoods[(i - 1, i - 1)] + \
                                    self.states[i - 1].log_next() + \
                                    self.states[i].output_likelihood(test[i])
        for i in range(1, self.model_size):
            for j in range(i + 1, len(test)):
                left = likelihoods[(i, j - 1)] + \
                       self.states[i].log_stay() + \
                       self.states[i].output_likelihood(test[j])
                down = likelihoods[(i - 1, j - 1)] + \
                       self.states[i - 1].log_next() + \
                       self.states[i].output_likelihood(test[j])
                likelihoods[(i, j)] = max(left, down)
        ret = likelihoods[(self.model_size - 1, len(test) - 1)]
        return ret
