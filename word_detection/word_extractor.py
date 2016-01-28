import numpy


class WordExtractor:
    def __init__(self, sample):
        """ Computes noise threshold using given sample """
        avg = numpy.average(numpy.absolute(sample))
        self.noise_threshold = avg

    def detect_words(self, rate, data, hint = 0):
        noise_array = self.__detect_noise(rate, data)
        self.__smooth_noise(rate, noise_array)
        return self.__split_words(rate, data, noise_array, hint)

    # detect sample as noise if average in previous or following 20/40/60ms
    # is above noise threshold
    # use dynamic programming to speed up the process
    # returns array of 0s and 1s depending if sample is classified as noise
    def __detect_noise(self, rate, data):
        noise = []
        window_size = 2 * rate / 100  # number of samples in 20ms
        n = len(data)
        dp_left = [[0, 0, 0] for x in range(0, n)]
        dp_left_len = [[0, 0, 0] for x in range(0, n)]
        dp_right = [[0, 0, 0] for x in range(0, n)]
        dp_right_len = [[0, 0, 0] for x in range(0, n)]
        for i in range(1, n):
            for j in range(0, 3):
                dp_left[i][j] = dp_left[i - 1][j] + numpy.absolute(data[i - 1])
                dp_left_len[i][j] = dp_left_len[i - 1][j] + 1
                if i - (j + 1) * window_size > 0:
                    dp_left[i][j] -= numpy.absolute(
                            data[i - (j + 1) * window_size - 1])
                    dp_left_len[i][j] -= 1
        for i in reversed(range(0, n - 1)):
            for j in range(0, 3):
                dp_right[i][j] = dp_right[i + 1][j] + numpy.absolute(
                        data[i + 1])
                dp_right_len[i][j] = dp_left_len[i + 1][j] + 1
                if i + (j + 1) * window_size < n - 1:
                    dp_right[i][j] -= numpy.absolute(
                            data[i + (j + 1) * window_size + 1])
                    dp_right_len[i][j] -= 1
        for i in range(0, n):
            if i > 0:
                avg_before = 0
                for j in range(0, 3):
                    avg_before = max(avg_before,
                            float(dp_left[i][j]) / float(dp_left_len[i][j]))
            else:
                avg_before = 0
            if i < n - 1:
                avg_after = 0
                for j in range(0, 3):
                    avg_after = max(avg_after,
                            float(dp_right[i][j]) / float(dp_right_len[i][j]))
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
    # if hint for number of words is passed, select that many words with
    # highest standard deviation or more if there is no significant drop
    def __split_words(self, rate, data, noise, hint):
        words = []
        min_word_length = rate / 10  # number of samples in 100ms
        last = []
        stds = []
        for i in range(0, len(data)):
            if noise[i] == 1:
                last.append(data[i])
                continue
            if len(last) >= min_word_length:
                words.append(last)
                stds.append(numpy.std(last))
            last = []
        if len(last) >= min_word_length:
            words.append(last)
            stds.append(numpy.std(last))
        if hint == 0:
            std_threshold = 0.25 * numpy.average(stds)
            return self.__filter_with_threshold(words, stds, std_threshold)
        else:
            return self.__filter_with_hint(words, stds, hint)

    # filter some words using computed standard deviation limit
    def __filter_with_threshold(self, words, stds, std_threshold):
        ret = []
        for word, std in zip(words, stds):
            if std > std_threshold:
                ret.append(word)
        return ret

    # filter using given hint for number of words
    # expect drop less than 10% to take more than hint
    def __filter_with_hint(self, words, stds, hint):
        stds.sort(reverse = True)
        if hint >= len(stds):
            return words
        limit = hint
        avg = numpy.average(stds[0 : hint])
        threshold = 0.95 * (stds[hint - 1] / avg)
        while limit < len(stds) and stds[limit] / avg > threshold:
            limit += 1
        if limit == len(stds):
            return words
        threshold = (stds[limit - 1] + stds[limit]) / 2.0
        ret = []
        for word, std in zip(words, stds):
            if std > threshold:
                ret.append(word)
        return ret
