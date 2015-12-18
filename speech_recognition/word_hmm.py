"""
    HMM modeling given word.
"""


class WordHmm:
    def __init__(self, word, coefficients_list):
        self.__word = word
        self.__build_hmm(coefficients_list)

    def __build_hmm(self, coefficients_list):
        pass
        # TODO: implement

    def word(self):
        return self.__word

    def try_matching(self, coefficients):
        return 0.0
        # TODO: implement