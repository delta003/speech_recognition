"""
    Library for detecting words' boundaries in speech.

        extract_words(rate, data)
    Splits data to multiple parts each representing one word
    input: sampling rate and samples, hint is expected number of words,
           but function can return more if it's unsure, 0 means no hint
    output: rate, list of parts, each part is word sample
"""


import numpy

from scipy.io import wavfile
from word_extractor import WordExtractor

__author__ = "Marko Bakovic (delta003)"


def extract_words(rate, data, hint = 0):
    # use first 100ms to set noise threshold
    samples = rate / 10  # number of samples in 100ms
    extractor = WordExtractor(data[:samples])
    words = extractor.detect_words(rate, data[samples:], hint)
    return rate, words
