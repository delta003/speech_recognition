"""
    Library for detecting words' boundaries in speech.

        extract_words(rate, data)
    Splits data to multiple parts each representing one word
    input: sampling rate and samples
    output: rate, list of parts, each part is word sample
"""


import numpy

from scipy.io import wavfile
from word_extractor import WordExtractor

__author__ = "Marko Bakovic (delta003)"


def extract_words(rate, data):
    # use first 100ms to set noise threshold
    samples = rate / 10  # number of samples in 100ms
    extractor = WordExtractor(data[:samples])
    print 'Extracting words...'
    words = extractor.detect_words(rate, data[samples:])
    print 'Done.'
    return rate, words


if __name__ == '__main__':
    rate, data = wavfile.read("../wav/hard0.wav")
    rate, words = extract_words(rate, data)
    for i in range(0, len(words)):
        filename = 'output/word' + str(i) + '.wav'
        wavfile.write(filename, rate, numpy.asarray(words[i]))
