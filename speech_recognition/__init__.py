"""
    Speaker independent speech recognition
    MFCC + HMM

        try_recognition(rate, data, recognizer)
    Match data sample to words from dictionary. Requires sampling rate
    same as dictionary sampling rate.
    input: sampling rate and samples, trained recognizer
    output: string with highest likelihood score and that likelihood
"""

from speech_recognizer import SpeechRecognizer

__author__ = "Marko Bakovic (delta003)"


def try_recognition(rate, data, recognizer):
    if rate != recognizer.rate:
        print('Sampling rate do NOT match dictionary sampling rate.')
        return '', -1.0
    word, likelihood = recognizer.search_word(data)
    return word, likelihood
