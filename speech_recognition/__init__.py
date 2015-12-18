"""
    Speaker independent speech recognition
    MFCC + HMM + dictionary

        try_recognition(rate, data)
    Match data sample to word based on dictionary. Requires sampling rate
    same as dictionary sampling rate.
    input: sampling rate and samples
    output: string with highest confidence score and that score, score is
        real number between 0 and 1
"""

from speech_recognition import SpeechRecognizer

__author__ = "Marko Bakovic (delta003)"


def try_recognition(rate, data, recognizer):
    if not recognizer.have_dictionary():
        recognizer.load_dictionary('dictionary/')
    if rate != recognizer.dictionary_sampling_rate():
        print('Sampling rate do NOT match dictionary sampling rate.')
        return '', -1.0
    print('Searching for match...')
    word, confidence = recognizer.search_word(rate, data)
    print('Done.')
    return word, confidence



if __name__ == '__main__':
    recognizer = SpeechRecognizer()
    recognizer.load_dictionary('dictionary/')