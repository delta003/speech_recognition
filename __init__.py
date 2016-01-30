import os
import string
import random
import warnings
import numpy
from scipy.io import wavfile
from speech_recognition import SpeechRecognizer, try_recognition
from word_detection import extract_words
from sound import record_wav, play_wav

__author__ = "Marko Bakovic (delta003)"


# generates random string
def id_generator(size = 10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# clears tmp folder
def clear_tmp():
    folder = 'tmp/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e


# helper method, outputs temporary wav file in tmp/ and returns file name
def save_tmp_wav(rate, speech):
    name = id_generator() + '.wav'
    wavfile.write('tmp/{0}'.format(name),
                  rate,
                  numpy.array(speech))
    return name


# helper method, saves wav in dictionary
def save_wav(word, name, rate, speech):
    if not os.path.isdir('dictionary/' + word):
        os.makedirs('dictionary/' + word)
    wavfile.write('dictionary/{0}/{1}'.format(word, name),
                  rate,
                  numpy.array(speech))


def main():
    warnings.simplefilter('ignore')
    clear_tmp()
    print('0. Simplified')
    print('1. Full')
    option = input('Select input: ')
    if option == 0:
        dir = 'simplified_dict/'
    else:
        dir = 'dictionary/'
    print('Loading recognizer, please wait...')
    speech_recognizer = SpeechRecognizer(dir)
    print('Loading completed.')
    print('0. Record new')
    print('1. Use existing wav file (from wav/)')
    print('2. Just record wav for later use')
    print('3. Split existing file (from wav/)')
    option = input('Select input: ')
    if option == 0:
        speech_file = record_wav()
    elif option == 1:
        speech_file = raw_input('File name (without .wav): ')
    elif option == 2:
        _ = record_wav()
        return
    elif option == 3:
        speech_file = raw_input('File name (without .wav): ')
        hint = input(
                'Hint maybe (enter expected number of words, 0 for no hint)? ')
        rate, samples = wavfile.read('wav/{0}.wav'.format(speech_file))
        rate, speech_words = extract_words(rate, samples, hint)
        for speech in speech_words:
            _ = save_tmp_wav(rate, speech)
        print('Extracted {0} words'.format(len(speech_words)))
        return
    else:
        print('Invalid option')
        return
    hint = input('Hint maybe (enter expected number of words, 0 for no hint)? ')
    rate, samples = wavfile.read('wav/{0}.wav'.format(speech_file))
    rate, speech_words = extract_words(rate, samples, hint)
    print('Extracted {0}'.format(len(speech_words)))
    correct_count = 0
    count = 0
    for speech in speech_words:
        speech_copy = speech[:]
        _ = raw_input('Press enter to continue...')
        wav_name = save_tmp_wav(rate, speech)
        play_wav('tmp/{0}'.format(wav_name))
        word, confidence = try_recognition(rate, speech_copy, speech_recognizer)
        print('Recognized [{0}] with confidence {1}'.format(word, confidence))
        count += 1
        correct = input('Is it correct (0/1)? ')
        if correct == 0:
            add = input('Do you want to add this word in dictionary (0/1)? ')
            if add == 0:
                print(':(')
            else:
                text_word = raw_input('Write string representation: ')
                text_word.rstrip('\n').replace(' ', '_')
                save_wav(text_word, wav_name, rate, speech)
                print('Added new word [{0}]'.format(text_word))
        else:
            correct_count += 1
            os.rename('tmp/{0}'.format(wav_name), 'tmp/{0}.wav'.format(word))
            print(':)')
    print('Recognized {0}% correctly.'.format(
            100.0 * float(correct_count) / float(count)))


if __name__ == '__main__':
    main()
