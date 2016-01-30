# trains model on random selected speakers and tests on the rest
# end-to-end test

import os
import random
import warnings
import scipy.io.wavfile as wavfile

from utility import id_generator, save_wav, reset_dir

from word_detection import extract_words
from speech_recognition import SpeechRecognizer, try_recognition

# constants
DIR_NAME = 'tests'
DICTIONARY = 'tests_dictionary'
TRAINING_SIZE = 6
SIMPLIFIED = False


def filter(word):
    if not SIMPLIFIED:
        return True
    return word in [
        'broj',
        'svoj',
        'kosa',
        'koliko',
        'razumem',
        'mislim',
        'postojim'
    ]


def train(dirs):
    print('Training on {0}'.format(dirs))
    reset_dir(DICTIONARY)
    for i, dir in enumerate(dirs):
        for file in os.walk(DIR_NAME + '/' + dir).next()[2]:
            word = file.split('_')[0]
            if not filter(word):
                continue
            rate, samples = wavfile.read('{0}/{1}/{2}'.format(
                    DIR_NAME, dir, file))
            _, speech_words = extract_words(rate, samples, 1)
            if len(speech_words) != 1:
                print('Error extracting {0}/{1}'.format(dir, file))
                continue
            wav_name = id_generator() + '.wav'
            save_wav(word, wav_name, rate, speech_words[0], DICTIONARY)
        print('Completed {0}%'.format(100.0 * (i + 1) / float(len(dirs))))
    print('Done training.')


def test(dirs):
    print('Testing on {0}'.format(dirs))
    print('Loading recognizer, please wait...')
    speech_recognizer = SpeechRecognizer(DICTIONARY)
    print('Loading completed.')
    all = 0
    correct = 0
    for i, dir in enumerate(dirs):
        curr_all = 0
        curr_correct = 0
        for file in os.walk(DIR_NAME + '/' + dir).next()[2]:
            expected_word = file.split('_')[0]
            if not filter(expected_word):
                continue
            rate, samples = wavfile.read('{0}/{1}/{2}'.format(
                    DIR_NAME, dir, file))
            rate, speech_words = extract_words(rate, samples, 1)
            if len(speech_words) != 1:
                print('Error extracting {0}/{1}'.format(dir, file))
                continue
            word, _ = try_recognition(rate, speech_words[0], speech_recognizer)
            all += 1
            curr_all += 1
            if word == expected_word:
                correct += 1
                curr_correct += 1
            else:
                print 'Expected {0}, got {1}'.format(expected_word, word)
        if curr_all > 0:
            print('Correct {0}% in {1}'.format(
                    100.0 * curr_correct / curr_all, dir))
        print('Completed {0}%'.format(100.0 * (i + 1) / float(len(dirs))))
    print('--------------------------------')
    print('Correct {0}%'.format(100.0 * correct / all))
    print('Done testing.')


# Results (Simplified):
#
# Training on ['_107', '_103', '_101', '_108', '_105', '_202']
# Testing on ['_201', '_203', '_106', '_102', '_205', '_104']
# Correct 64.2857142857%
#
# Training on ['_109', '_101', '_110', '_107', '_106', '_102']
# Testing on ['_205', '_105', '_103', '_202', '_108', '_203', '_104', '_201']
# Correct 51.7857142857%
#
# Training on ['_101', '_102', '_103', '_104', '_107']
# Testing on ['_202', '_108', '_105', '_205', '_106', '_203', '_201']
# Correct 61.2244897959%
#
# Training on ['_101', '_102', '_103', '_104', '_107']
# Testing on ['_105', '_106', '_108', '_201', '_202', '_205']
# Correct 66.6666666667%
def integration_test():
    print('0. Simplified')
    print('1. Full')
    option = input('Select input: ')
    if option == 0:
        global SIMPLIFIED
        SIMPLIFIED = True
    warnings.simplefilter('ignore')
    subdirs = os.walk(DIR_NAME).next()[1]
    random.shuffle(subdirs)
    train_d = subdirs[:TRAINING_SIZE]
    test_d = subdirs[TRAINING_SIZE:]
    train(train_d)
    test(test_d)


if __name__ == '__main__':
    integration_test()
