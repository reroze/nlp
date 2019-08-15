#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from word2vec import *
from sgd import *

# Check Python Version检查python版本
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results 设置随机种子
random.seed(314)
dataset = StanfordSentiment()#导入数据集
tokens = dataset.tokens()#拿token
nWords = len(tokens)#words的个数

#print("dataset:",dataset)
#print("tokens:",tokens)#tokens是对应的单词和对应的矩阵序列号{'the': 0, 'rock': 1, 'is': 2, 'destined': 3, 'to': 4, 'be': 5, '21st'....这种之类的
#print("nWords",nWords)#一共19539个


# We are going to train 10-dimensional vectors for this assignment 这一次训练10个向量
dimVectors = 10

# Context size 滑动框
C = 5

# Reset the random seed to make sure that everyone gets the same results#对应的np的随机种子
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
wordVectors = np.concatenate(#第一个参数是一个根据nWords和dimVector生成的随机数，第二个参数是是一个对应的【nWords*dimVectors】的矩阵，第三个参数是？
    ((np.random.rand(nWords, dimVectors) - 0.5) /
       dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingLossAndGradient),
    wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.在训练的时候再进行泛化

print("sanity check: cost at convergence should be around or below 10")#健康状态检测
print("training took %d seconds" % (time.time() - startTime))

# concatenate the input and output word vectors
wordVectors = np.concatenate(
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),
    axis=0)

visualizeWords = [
    "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
    "annoying", "female", "male", "queen", "king", "man", "woman", "rain", "snow",
    "hail", "coffee", "tea"]

visualizeIdx = [tokens[word] for word in visualizeWords]
#print("visuallizeIdx:",visualizeIdx)
##################################################################################################################

visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')
