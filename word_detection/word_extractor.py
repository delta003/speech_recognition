import numpy


class WordExtractor:
    def __init__(self, sample):
        """ Computes silence threshold using given sample """
        avg = numpy.average(numpy.absolute(sample))
        std = numpy.std(sample)
        self.noise_threshold = avg
        self.std_threshold = std

    def detect_words(self, rate, data):
        noise_array = self.__detect_noise(rate, data)
        self.__smooth_noise(rate, noise_array)
        return self.__split_words(rate, data, noise_array)

    # detect sample as noise if average in previous or following 20/40/60ms
    # is above noise threshold
    # returns array of 0s and 1s depending if sample is classified as noise
    def __detect_noise(self, rate, data):
        noise = []
        window_size = 2 * rate / 100  # number of samples in 20ms
        n = len(data)
        for i in range(0, n):
            if i > 0:
                avg_before = 0
                for j in range(0, 3):
                    avg_before = max(avg_before, numpy.average(numpy.absolute(
                        data[max(0, i - (j + 1) * window_size) : i])))
            else:
                avg_before = 0
            if i < n - 1:
                avg_after = 0
                for j in range(0, 3):
                    avg_after = max(avg_after, numpy.average(numpy.absolute(
                        data[i + 1 : min(n, i + (j + 1) * window_size)])))
            else:
                avg_after = 0
            if max(avg_before, avg_after) > self.noise_threshold:
                noise.append(1)
            else:
                noise.append(0)
        return noise

    # performs smoothing of noise array, decides to switch from silence to
    # noise for some interval smaller then 200ms if there is at least 3 times
    # larger noise interval before or after and at least the same on the other
    # side
    def __smooth_noise(self, rate, noise):
        window_size = 2 * rate / 10  # number of samples in 200ms
        n = len(noise)
        i = 0
        while i < n:
            if noise[i] == 0:
                j = i
                while j + 1 < n and noise[j + 1] == 0:
                    j += 1
                if j - i + 1 < window_size:
                    left_i = i - 1
                    while left_i - 1 >= 0 and noise[left_i - 1] == 1:
                        left_i -= 1
                    right_j = j + 1
                    while right_j + 1 < n and noise[right_j + 1] == 1:
                        right_j += 1
                    this_len = j - i + 1
                    left_len = i - left_i
                    right_len = right_j - j
                    if (min(left_len, right_len) >= this_len and
                        max(left_len, right_len) >= 3 * this_len):
                        for k in range(i, j + 1):
                            noise[k] = 1
                    i = right_j + 1
                else:
                    i = j + 1
            else:
                i += 1

    # extract words depending on array of 0s and 1s
    # word must be longer then 100ms
    def __split_words(self, rate, data, noise):
        words = []
        min_word_length = rate / 10  # number of samples in 100ms
        last = []
        for i in range(0, len(data)):
            if noise[i] == 1:
                last.append(data[i])
                continue
            if len(last) >= min_word_length:
                words.append(last)
            last = []
        if len(last) >= min_word_length:
            words.append(last)
        return words
