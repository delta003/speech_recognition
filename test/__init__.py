# Run test for evaluation

import json
import warnings
import scipy.io.wavfile as wavfile

from utility import id_generator, save_wav

from word_detection import extract_words
from speech_recognition import SpeechRecognizer, try_recognition


def train(evaluation):
    n = len(evaluation['training'])
    for i, example in enumerate(evaluation['training']):
        example_name = example['file']
        hint = example['expected_words_count']
        rate, samples = wavfile.read('../wav/{0}.wav'.format(example_name))
        rate, speech_words = extract_words(rate, samples, hint)
        if len(speech_words) != len(example['words']):
            'Error during training for {0}'.format(example_name)
            continue
        for word, speech in zip(example['words'], speech_words):
            wav_name = id_generator() + '.wav'
            save_wav(word, wav_name, rate, speech, '../dictionary')
        print('Completed {0}%'.format(100.0 * (i + 1) / float(n)))


def test(evaluation):
    print('Loading recognizer, please wait...')
    speech_recognizer = SpeechRecognizer('../dictionary/')
    print('Loading completed.')
    n = len(evaluation['test'])
    all = 0
    correct = 0
    for i, example in enumerate(evaluation['test']):
        curr_all = 0
        curr_correct = 0
        example_name = example['file']
        hint = example['expected_words_count']
        rate, samples = wavfile.read('../wav/{0}.wav'.format(example_name))
        rate, speech_words = extract_words(rate, samples, hint)
        if len(speech_words) != len(example['words']):
            print('Error in extracting words. Skipping {0}.'.format(
                    example_name))
            continue
        for word, sol_word in zip(speech_words, example['words']):
            sol, _ = try_recognition(rate, word, speech_recognizer)
            #print('Expected {0}, recognized {1}'.format(sol_word, sol))
            all += 1
            curr_all += 1
            if sol == sol_word:
                correct += 1
                curr_correct += 1
        print('Completed {0}%'.format(100.0 * (i + 1) / float(n)))
        print('Stats on {0}, correct {1}%'.format(
                example_name,
                100.0 * curr_correct / curr_all))
        break
    print('--------------------------------------')
    print('Correct {0}%'.format(100.0 * correct / all))


def test_word_extractor(evaluation):
    n = len(evaluation['examples20'])
    all = 0
    correct = 0
    for i, test in enumerate(evaluation['examples20']):
        test_name = test['file']
        hint = test['expected_words_count']
        rate, samples = wavfile.read('../wav/{0}.wav'.format(test_name))
        _, speech_words = extract_words(rate, samples, hint)
        #_, speech_words_no_hint = extract_words(rate, samples)
        speech_words_no_hint = []
        print('Test {0}: expected = {1}, extracted = {2}, with no hint = {3}'
            .format(test_name,
                    test['expected_words_count'],
                    len(speech_words),
                    len(speech_words_no_hint)))
        all += 1
        if len(speech_words) == test['expected_words_count']:
            correct += 1
        print('Completed {0}%'.format(100.0 * (i + 1) / float(n)))
    print('Extracted {0}% correctly.'.format(
            100.0 * float(correct) / float(all)))


def main():
    warnings.simplefilter('ignore')
    with open('test.json') as test_file:
        evaluation = json.load(test_file)
        test_file.close()
    print('0. Test word extractor')
    print('1. Train')
    print('2. Test')
    option = input('Select: ')
    if option == 0:
        test_word_extractor(evaluation)
    elif option == 1:
        train(evaluation)
    elif option == 2:
        test(evaluation)


if __name__ == '__main__':
    main()