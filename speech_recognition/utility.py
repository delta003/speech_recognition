import numpy as np
from sklearn.mixture import log_multivariate_normal_density


# finds mean
# https://en.wikipedia.org/wiki/Sample_mean_and_covariance
def mean_vector(list):
    ret = [0.0 for l in list[0]]
    n = len(ret)
    for l in list:
        assert n == len(l)
        for j in range(0, n):
            ret[j] += l[j]
    for i in range(0, n):
        ret[i] /= float(len(list))
    return ret


# computes covariance
# https://en.wikipedia.org/wiki/Sample_mean_and_covariance
def covariance_matrix(mean, list):
    n = len(mean)
    covariance = [[0.0 for i in range(0, n)] for j in range(0, n)]
    variance = [0.0 for i in range(0, n)]
    for vector in list:
        for i in range(0, n):
            variance[i] += np.square(mean[i] - vector[i])
    for i in range(0, n):
        variance[i] /= float(len(list) - 1)
        covariance[i][i] = variance[i]
    return covariance


# computes Gaussian emission log likelihood
def gaussian_log_likelihood(x, mean, covariance):
    covariance_diag = []
    for i in range(0, len(mean)):
        covariance_diag.append(covariance[i][i])
    likelihood = log_multivariate_normal_density(np.array([x]),
                                                 np.array([mean]),
                                                 np.array([covariance_diag]),
                                                 covariance_type = 'diag')
    return likelihood[0][0]


# splits the list into evenly sized chunks (last one can be larger)
def chunks(list, count):
    ret = []
    chunk_len = len(list) / count
    add = len(list) % count
    last = 0
    for i in range(0, count):
        chunk = []
        for j in range(0, chunk_len):
            chunk.append(list[last])
            last += 1
        if i < add:
            chunk.append(list[last])
            last += 1
        ret.append(chunk)
    return ret


# checks if two vectors are equal using 0.001 as equality threshold
def float_vector_equal(vector1, vector2):
    EPS = 0.001
    if len(vector1) != len(vector2):
        return False
    for i in range(0, len(vector1)):
        if abs(vector1[i] - vector2[i]) > EPS:
            return False
    return True

