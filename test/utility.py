import os
import string
import random
import numpy
import shutil
import scipy.io.wavfile as wavfile


# generates random string
def id_generator(size = 10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# helper method, saves wav in dictionary
def save_wav(word, name, rate, speech, path):
    if not os.path.isdir(path + '/' + word):
        os.makedirs(path + '/' + word)
    wavfile.write('{0}/{1}/{2}'.format(path, word, name),
                  rate,
                  numpy.array(speech))


def reset_dir(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

